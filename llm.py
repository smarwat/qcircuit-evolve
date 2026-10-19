"""Model clients for the orchestrator.

ClaudeClient: streamed Messages API calls with prompt caching on the system prompt
and task spec, token/cost accounting from `usage`, and retries with backoff.
MockClient: deterministic stand-in with the same interface (no network).

API shapes used here were checked against anthropic-sdk-python 1.15.0
(types/message_create_params.py, usage.py, thinking_config_*.py,
output_config_param.py, cache_control_ephemeral_param.py) and the docs at
platform.claude.com (models overview, pricing, prompt caching, thinking, effort).
"""

import random
import re
import time

RETRYABLE_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}


class FatalAPIError(Exception):
    """A non-retryable API error (bad request, auth, ...). The run should stop."""


def usage_to_dict(u):
    """Normalise SDK `Usage` (or a dict) into flat token counts."""
    get = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
    cc = get("cache_creation")
    if cc is not None and not isinstance(cc, dict):
        cc = {"ephemeral_5m_input_tokens": getattr(cc, "ephemeral_5m_input_tokens", 0) or 0,
              "ephemeral_1h_input_tokens": getattr(cc, "ephemeral_1h_input_tokens", 0) or 0}
    write = get("cache_creation_input_tokens") or 0
    return {
        "input": get("input_tokens") or 0,          # tokens after the last cache breakpoint
        "output": get("output_tokens") or 0,        # includes thinking
        "cache_read": get("cache_read_input_tokens") or 0,
        "cache_write": write,
        "cache_write_5m": (cc or {}).get("ephemeral_5m_input_tokens", 0) or 0,
        "cache_write_1h": (cc or {}).get("ephemeral_1h_input_tokens", 0) or 0,
    }


def cost_usd(tok, prices, cache_ttl="5m"):
    """Dollar cost of one call. Cache writes are split by TTL when the API reports
    the breakdown; otherwise all writes are priced at the configured TTL."""
    w5, w1 = tok.get("cache_write_5m", 0), tok.get("cache_write_1h", 0)
    rest = max(tok.get("cache_write", 0) - w5 - w1, 0)
    if cache_ttl == "1h":
        w1 += rest
    else:
        w5 += rest
    total = (tok["input"] * prices["input"] + tok["output"] * prices["output"]
             + tok["cache_read"] * prices["cache_read"]
             + w5 * prices["cache_write_5m"] + w1 * prices["cache_write_1h"])
    return total / 1e6


def _retry_delay(attempt, base, cap, err=None):
    delay = min(cap, base * (2 ** attempt)) * (0.5 + random.random() / 2)
    resp = getattr(err, "response", None)
    try:
        ra = float(resp.headers.get("retry-after")) if resp is not None else None
    except (TypeError, ValueError, AttributeError):
        ra = None
    return max(delay, ra or 0.0)


class ClaudeClient:
    def __init__(self, llm_cfg, system_rules, task_spec):
        import anthropic  # lazy: --mock never needs the SDK

        self.anthropic = anthropic
        self.cfg = llm_cfg
        self.client = anthropic.Anthropic(max_retries=2)
        cache_control = {"type": "ephemeral", "ttl": llm_cfg["cache_ttl"]}
        # Both blocks are static for the whole run; one breakpoint on the last
        # caches the system prompt and the task spec together.
        self.system = [
            {"type": "text", "text": system_rules},
            {"type": "text", "text": task_spec, "cache_control": cache_control},
        ]

    def model_info(self):
        """Ask the Models API for the model's limits/capabilities (None on failure)."""
        try:
            m = self.client.models.retrieve(self.cfg["model"])
        except Exception as e:  # noqa: BLE001 - informational only
            return {"error": f"{type(e).__name__}: {e}"}
        caps = getattr(m, "capabilities", None)
        eff = getattr(caps, "effort", None) if caps else None
        info = {"id": m.id, "max_tokens": getattr(m, "max_tokens", None),
                "max_input_tokens": getattr(m, "max_input_tokens", None)}
        if eff is not None:
            info["effort_levels"] = [lvl for lvl in ("low", "medium", "high", "xhigh", "max")
                                     if getattr(getattr(eff, lvl, None), "supported", False)]
        return info

    def _retryable(self, e):
        a = self.anthropic
        if isinstance(e, a.APIConnectionError):
            return True
        if isinstance(e, a.APIStatusError):
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                return True
            body = e.body if isinstance(e.body, dict) else {}
            etype = (body.get("error") or {}).get("type") if isinstance(body.get("error"), dict) else None
            return etype in RETRYABLE_ERROR_TYPES
        return False

    def call(self, user_prompt, on_first_event=None):
        """One streamed request. Returns text, token usage, cost, stop reason, timing."""
        c = self.cfg
        params = dict(
            model=c["model"],
            max_tokens=c["max_tokens"],
            system=self.system,
            messages=[{"role": "user", "content": user_prompt}],
            thinking=c["thinking"],
            output_config={"effort": c["effort"]},
        )
        t0 = time.monotonic()
        last = None
        for attempt in range(c["max_retries"] + 1):
            try:
                with self.client.messages.stream(**params) as stream:
                    for _ in stream:
                        if on_first_event is not None:
                            on_first_event()
                            on_first_event = None
                    msg = stream.get_final_message()
                tok = usage_to_dict(msg.usage)
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                thinking = "".join(getattr(b, "thinking", "") or "" for b in msg.content
                                   if getattr(b, "type", "") == "thinking")
                return {
                    "text": text,
                    "thinking_summary": thinking,
                    "usage": tok,
                    "cost_usd": cost_usd(tok, c["prices_per_mtok"], c["cache_ttl"]),
                    "stop_reason": msg.stop_reason,
                    "api_s": round(time.monotonic() - t0, 2),
                    "attempts": attempt + 1,
                    "request_id": getattr(msg, "_request_id", None),
                    "model": msg.model,
                }
            except Exception as e:  # noqa: BLE001
                # Note: tokens streamed before a mid-stream failure may be billed but
                # are not reported in usage, so cost tracking can undercount retries.
                if not self._retryable(e):
                    raise FatalAPIError(f"{type(e).__name__}: {e}") from e
                last = e
                if attempt < c["max_retries"]:
                    time.sleep(_retry_delay(attempt, c["retry_base_delay"], c["retry_max_delay"], e))
        raise FatalAPIError(f"gave up after {c['max_retries'] + 1} attempts: {type(last).__name__}: {last}")


class MockClient:
    """Deterministic fake model. Cycles through reply shapes so every orchestrator
    path runs without the API:
      diff_hidden_cheat  cheaper at n=32, wrong only at a hidden width -> fails verify
      diff_improve       a real improvement (binary Euclid needs only 2n-1 steps) -> new best
      diff_missing       SEARCH text absent -> edit rejected
      diff_broken        applies, but drops the uncompute -> invalid
      full_redesign      full program (the divsteps seed)
      full_same          full program, the parent plus a comment
      diff_ok            correct but wasteful (cancelling Toffoli pair)"""

    KINDS = ["diff_hidden_cheat", "diff_improve", "diff_missing", "diff_broken",
             "full_redesign", "full_same", "diff_ok"]
    LOOP = "    for it in range(2 * n):\n"

    def __init__(self, llm_cfg, redesign_source=None):
        self.cfg = llm_cfg
        self.calls = 0
        self.redesign_source = redesign_source

    def model_info(self):
        return {"id": "mock", "max_tokens": None}

    def call(self, user_prompt, on_first_event=None):
        if on_first_event is not None:
            on_first_event()
        k = self.KINDS[self.calls % len(self.KINDS)]
        self.calls += 1
        m = re.search(r"<parent>\n```python\n(.*?)```\n</parent>", user_prompt, re.S)
        parent = m.group(1) if m else ""
        last_line = parent.rstrip("\n").split("\n")[-1] + "\n"
        if k == "diff_improve":
            text = ("Mock: run 2n-1 iterations.\n<<<<<<< SEARCH\n" + self.LOOP
                    + "=======\n    for it in range(2 * n - 1):\n>>>>>>> REPLACE\n")
        elif k == "diff_hidden_cheat":
            text = ("Mock: fewer iterations, wrong only at a hidden width.\n<<<<<<< SEARCH\n" + self.LOOP
                    + "=======\n    for it in range(2 * n - 1 if n != -1 else 2 * n - 4):\n>>>>>>> REPLACE\n")
        elif k == "full_same":
            text = f"Mock: resubmit the parent.\n```python\n# mock {self.calls}\n{parent}```\n"
        elif k == "diff_ok":  # correct but wasteful: a cancelling Toffoli pair
            text = ("Mock: diff that appends a cancelling Toffoli pair.\n<<<<<<< SEARCH\n" + last_line
                    + "=======\n" + last_line
                    + "    c.ccx(c.inp[0], c.inp[1], c.anc[-1])\n    c.ccx(c.inp[0], c.inp[1], c.anc[-1])\n"
                    + ">>>>>>> REPLACE\n")
        elif k == "diff_missing":
            text = ("Mock: diff against text that is not in the parent.\n<<<<<<< SEARCH\n"
                    "    this_line_is_not_in_the_parent()\n=======\n    pass\n>>>>>>> REPLACE\n")
        elif k == "diff_broken":  # applies cleanly, but skips the uncompute
            text = ("Mock: diff that drops the uncompute.\n<<<<<<< SEARCH\n" + last_line
                    + "=======\n    pass  # mock: uncompute removed\n>>>>>>> REPLACE\n")
        else:
            src = self.redesign_source or parent
            text = f"Mock: redesign.\n```python\n{src}```\n"
        out_tokens = len(text) // 4
        in_tokens = len(user_prompt) // 4
        tok = {"input": in_tokens, "output": out_tokens,
               "cache_read": 2000 if self.calls > 1 else 0,
               "cache_write": 0 if self.calls > 1 else 2000,
               "cache_write_5m": 0, "cache_write_1h": 0}
        return {"text": text, "thinking_summary": "", "usage": tok,
                "cost_usd": cost_usd(tok, self.cfg["prices_per_mtok"], self.cfg["cache_ttl"]),
                "stop_reason": "end_turn", "api_s": 0.0, "attempts": 1, "request_id": None,
                "model": "mock"}
