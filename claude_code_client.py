"""Backend that runs the model through the user's Claude Code subscription.

Each call is one headless subprocess:

    claude -p --output-format json --model claude-opus-5-5 --effort high \
           --tools "" --max-turns 1 --no-session-persistence --system-prompt <rules+spec>

with the user prompt on stdin, run from a fresh empty temp directory (so it cannot
see this repo, in particular config.py with the hidden seeds and primes) and with an
allowlisted environment (no ANTHROPIC_* variables, no parent-session CLAUDE_* ones),
so it can only authenticate with the subscription login.

Flags were checked against `claude --help` (Claude Code 2.1.157). The JSON result
looks like {"type": "result", "subtype": "success", "is_error": bool,
"api_error_status": int|None, "result": str, "num_turns": int, "duration_ms": int,
"duration_api_ms": int, "total_cost_usd": float, "usage": {...},
"modelUsage": {model: {...}}, "session_id": str, ...}. Note that a failed call can
still say subtype "success": `is_error` is what matters.
"""

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from llm import FatalAPIError, usage_to_dict


class ClaudeCodeError(Exception):
    """A failed call that should be logged as a failed attempt (the run continues)."""


class CLIAuthError(FatalAPIError):
    """The CLI is not authenticated; the run must stop."""


class UsageLimitError(Exception):
    """Subscription usage limit or rate limit. `reset_at` is a Unix time or None."""

    def __init__(self, message, reset_at=None):
        super().__init__(message)
        self.reset_at = reset_at


_LIMIT_RE = re.compile(r"usage limit|limit reached|hit your limit|limit will reset|rate[ _-]?limit|"
                       r"too many requests|out of (?:extra )?usage|quota", re.I)
_AUTH_RE = re.compile(r"failed to authenticate|not logged in|please run /login|invalid api key|"
                      r"oauth (?:access )?token|authentication", re.I)
_UNSUPPORTED_RE = re.compile(r"does not support this model|or newer is required|unrecognized.?model|"
                             r"model.{0,40}not (?:found|available|supported)|invalid model", re.I)
_TRANSIENT_RE = re.compile(r"overloaded|internal server error|api_error|\b5\d\d\b|timed? ?out|"
                           r"connection (?:error|reset|refused)|ECONNRESET|socket hang up", re.I)


def parse_reset_time(text, now=None):
    """Best-effort reset time from a CLI limit message; returns Unix time or None.

    Handles "...|1760000000" (epoch suffix), "resets 3pm", "resets at 3:30 pm
    (America/Los_Angeles)", "resets 15:00", "try again in 25 minutes", and ISO
    timestamps. Anything else returns None (the caller then exits for --resume)."""
    now = time.time() if now is None else now
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:try again|retry|resets?) in (\d+)\s*(second|sec|minute|min|hour|hr)s?", text, re.I)
    if m:
        mult = {"s": 1, "m": 60, "h": 3600}[m.group(2)[0].lower()]
        return now + int(m.group(1)) * mult
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", text)
    if m:
        try:
            t = dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.astimezone()
            return t.timestamp()
        except ValueError:
            pass
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:\(([^)]+)\))?", text, re.I)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            return None
        tz = None
        if m.group(4):
            try:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(m.group(4).strip())
            except Exception:  # noqa: BLE001 - unknown zone name: fall back to local time
                tz = None
        base = dt.datetime.fromtimestamp(now, tz) if tz else dt.datetime.fromtimestamp(now).astimezone()
        t = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if t.timestamp() <= now:
            t += dt.timedelta(days=1)
        return t.timestamp()
    return None


def clean_env(allowlist, environ=None):
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k in allowlist if k in environ}


def _answering_model(model_usage):
    """(main model, all models) from the result's modelUsage map."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None, []

    def out_tokens(v):
        if not isinstance(v, dict):
            return 0
        return v.get("outputTokens") or v.get("output_tokens") or 0

    models = sorted(model_usage)
    main = max(models, key=lambda m: out_tokens(model_usage[m]))
    return main, models


class ClaudeCodeClient:
    def __init__(self, cc_cfg, system_prompt, warn=None):
        self.cfg = cc_cfg
        self.system_prompt = system_prompt
        self.warn = warn or (lambda msg: print(msg, file=sys.stderr, flush=True))

    # `--tools ""` only removes built-in tools. MCP servers (including claude.ai connectors),
    # plugins, hooks and skills still load unless these are passed too: --restricted ignores
    # user/project/local settings, --strict-mcp-config with an empty --mcp-config loads no
    # MCP servers, --disable-slash-commands drops skills. Measured: without them the model
    # saw the user's connector tools (~13K-147K context tokens); with them, none (~600).
    ISOLATION = ["--restricted", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                 "--disable-slash-commands"]
    REQUIRED_FLAGS = ["--output-format", "--model", "--effort", "--tools", "--no-session-persistence",
                      "--system-prompt", "--restricted", "--strict-mcp-config", "--mcp-config",
                      "--disable-slash-commands"]  # --max-turns works but is not listed in --help

    def command(self):
        c = self.cfg
        return [c["binary"], "-p", "--output-format", "json", "--model", c["model"],
                "--effort", c["effort"], "--tools", "", "--max-turns", str(c["max_turns"]),
                *self.ISOLATION, "--no-session-persistence", "--system-prompt", self.system_prompt]

    def check_cli(self):
        """Fail fast unless the CLI supports every flag the isolated call needs."""
        try:
            proc = subprocess.run([self.cfg["binary"], "--help"], capture_output=True, text=True, timeout=60,
                                  env=clean_env(self.cfg["env_allowlist"]), cwd=tempfile.gettempdir())
            ver = subprocess.run([self.cfg["binary"], "--version"], capture_output=True, text=True, timeout=60,
                                 env=clean_env(self.cfg["env_allowlist"]), cwd=tempfile.gettempdir())
        except (OSError, subprocess.TimeoutExpired) as e:
            raise FatalAPIError(f"cannot run `{self.cfg['binary']}`: {e}") from None
        missing = [f for f in self.REQUIRED_FLAGS if not re.search(re.escape(f) + r"\b", proc.stdout)]
        version = (ver.stdout or "").strip()
        if missing:
            raise FatalAPIError(f"`{self.cfg['binary']}` ({version or 'unknown version'}) lacks {missing}, "
                                "needed to run the model with no tools, MCP servers, plugins or skills. "
                                "Update it (`claude update`) or pass --claude-bin pointing at a newer CLI.")
        return version

    def model_info(self):
        return {"backend": "claude-code", "binary": shutil.which(self.cfg["binary"]),
                "model": self.cfg["model"], "effort": self.cfg["effort"]}

    def _run_once(self, user_prompt):
        workdir = tempfile.mkdtemp(prefix="qce-claude-")
        try:
            t0 = time.monotonic()
            try:
                proc = subprocess.run(self.command(), input=user_prompt, capture_output=True, text=True,
                                      cwd=workdir, env=clean_env(self.cfg["env_allowlist"]),
                                      timeout=self.cfg["timeout_s"])
            except subprocess.TimeoutExpired:
                raise ClaudeCodeError(f"claude -p timed out after {self.cfg['timeout_s']}s") from None
            except FileNotFoundError:
                raise FatalAPIError(f"`{self.cfg['binary']}` not found on PATH") from None
            wall = time.monotonic() - t0
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return proc, wall

    def _classify(self, text, status=None):
        if status == 429 or _LIMIT_RE.search(text or ""):
            return "limit"
        if status in (401, 403) or _AUTH_RE.search(text or ""):
            return "auth"
        if _UNSUPPORTED_RE.search(text or ""):
            return "unsupported"
        if (status is not None and status >= 500) or _TRANSIENT_RE.search(text or ""):
            return "transient"
        return "other"

    def call(self, user_prompt, on_first_event=None):
        if on_first_event is not None:
            on_first_event()  # the CLI manages its own prompt cache; nothing to stagger
        c = self.cfg
        for attempt in range(c["max_retries"] + 1):
            proc, wall = self._run_once(user_prompt)
            out, err = proc.stdout or "", proc.stderr or ""
            try:
                data = json.loads(out)
                if not isinstance(data, dict):
                    raise ValueError("not an object")
            except ValueError:
                blob = (err + "\n" + out).strip()
                kind = self._classify(blob)
                if kind == "limit":
                    raise UsageLimitError(blob[:500], parse_reset_time(blob))
                if kind == "auth":
                    raise CLIAuthError(f"claude CLI is not authenticated: {blob[:300]}. "
                                       "Run `claude auth login` in a terminal, then --resume.")
                if kind == "transient" and attempt < c["max_retries"]:
                    time.sleep(min(c["retry_max_delay"], c["retry_base_delay"] * 2 ** attempt))
                    continue
                raise ClaudeCodeError(f"malformed output from claude -p (exit {proc.returncode}): "
                                      f"{blob[:300]!r}")
            text = data.get("result")
            if data.get("is_error") or not isinstance(text, str):
                status = data.get("api_error_status")
                msg = str(text if text is not None else data.get("subtype"))
                kind = self._classify(msg, status)
                if kind == "limit":
                    raise UsageLimitError(msg[:500], parse_reset_time(msg))
                if kind == "auth":
                    raise CLIAuthError(f"claude CLI authentication failed ({status}): {msg[:300]}. "
                                       "Run `claude auth login` in a terminal, then --resume.")
                if kind == "unsupported":
                    raise FatalAPIError(f"the claude CLI cannot use model {c['model']!r}: {msg[:300]} "
                                        "(update the CLI, or point --claude-bin at a newer one, then --resume)")
                if kind == "transient" and attempt < c["max_retries"]:
                    time.sleep(min(c["retry_max_delay"], c["retry_base_delay"] * 2 ** attempt))
                    continue
                raise ClaudeCodeError(f"claude -p returned an error (status {status}): {msg[:300]}")

            main, models = _answering_model(data.get("modelUsage"))
            expected = c["expected_model"]
            mismatch = main is None or expected not in main
            if mismatch:
                self.warn("!" * 72 + f"\nWARNING: expected {expected} but the CLI reported "
                          f"{main or 'no model'} (all models: {models}).\n" + "!" * 72)
            usage = usage_to_dict(data.get("usage") or {})
            denials = data.get("permission_denials") or []
            if (data.get("num_turns") or 1) > c["max_turns"] or denials:
                self.warn(f"WARNING: claude -p reported num_turns={data.get('num_turns')} "
                          f"permission_denials={denials}: the model tried to act, not just answer.")
            return {
                "text": text,
                "thinking_summary": "",
                "usage": usage,
                "cost_usd": 0.0,  # subscription: not billed per call
                "notional_cost_usd": data.get("total_cost_usd"),
                "stop_reason": data.get("stop_reason"),
                "api_s": round(wall, 2),
                "attempts": attempt + 1,
                "request_id": data.get("session_id"),
                "model": main,
                "models_used": models,
                "model_mismatch": mismatch,
                "num_turns": data.get("num_turns"),
                "permission_denials": denials,
                "context_tokens": usage["input"] + usage["cache_read"] + usage["cache_write"],
                "duration_ms": data.get("duration_ms"),
                "duration_api_ms": data.get("duration_api_ms"),
            }
        raise ClaudeCodeError("claude -p kept failing with transient errors")
