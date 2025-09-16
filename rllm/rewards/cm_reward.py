import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rllm.agents.context_manager_agent import _format_verifier_results
from rllm.rewards.code_reward import (
    extract_code_from_model,
    clean_code_main_block,
    lcb_check_correctness_v2,
    taco_to_lcb_format,
    leetcode_check_correctness,
    kodcode_check_correctness,
    humanevalplus_check_correctness,
)
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType

# -------------------------------
# Optional Together Code Tool
# -------------------------------
try:
    from rllm.tools.code_tools.together_tool import TogetherCodeTool  # optional
    _HAS_TCI = True
except Exception:
    _HAS_TCI = False


def _build_chat_url(base_url: str) -> str:
    base = base_url.rstrip('/')
    # if base already ends with /v{number}, don't add another /v1
    if re.search(r'/v\d+$', base):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


_SESSION = requests.Session()
_RETRY = Retry(
    total=3, connect=3, read=3,
    backoff_factor=1.5,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["POST"])
)
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))


# -------------------------------
# prompt builder (keeps old behavior)
# -------------------------------
def build_solver_prompt(
    problem: str,
    use_solver_cot: bool,
    feedback: Optional[str],
    prev_attempts: Optional[List[Dict[str, Any]]],
) -> str:
    parts = [problem or ""]
    if prev_attempts:
        last = prev_attempts[-1]
        formatted = _format_verifier_results(last.get("verifier_results", {}))
        if not use_solver_cot:
            last_attempt_txt = last.get("solver_output") or ""
        else:
            last_attempt_txt = last.get("solver_full_output") or ""

        parts.append("\nPrevious attempt (for reference; DO NOT copy):\n")
        parts.append(last_attempt_txt.strip())
        parts.append("\nUnit test results summary:\n")
        parts.append(formatted.strip())
    if feedback and feedback.strip():
        parts.append(
            "\nYour previous attempt was incorrect. For your next solution, apply the following guidance:\n"
        )
        parts.append(feedback.strip())
    return "\n".join(parts)


# -------------------------------
# HTTP chat (non-streaming)
# -------------------------------

def _http_chat_once(
    base_url: str,
    model: str,
    prompt: str,
    api_key: Optional[str] = None,
    gen_cfg: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
) -> str:
    # Per-call session with retry (safe drop-in; you can hoist to a module-global Session if you prefer)
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "Content-Type": "application/json",
        "Connection": "close",  # avoid flaky long-lived keep-alive sockets
    }
    if api_key and api_key not in ("EMPTY", "None", None):
        headers["Authorization"] = f"Bearer {api_key}"

    # Normalize generation params & keep response predictable
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # single JSON (no SSE)
        "n": 1,
    }
    if gen_cfg:
        norm = dict(gen_cfg)
        if "max_new_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_new_tokens")
        # Only include non-None values
        for k, v in norm.items():
            if v is not None:
                payload[k] = v
    # Sensible defaults if not provided
    payload.setdefault("temperature", 0.0)
    payload.setdefault("max_tokens", 1024)

    url = _build_chat_url(base_url)
    try:
        # Separate connect/read timeouts helps diagnose slow servers
        resp = session.post(url, headers=headers, json=payload, timeout=(600.0, timeout_s))
        resp.raise_for_status()
        data = resp.json()  # can raise ValueError if body is truncated/invalid
    except Exception:
        # Let caller's retry/backoff wrapper handle this
        raise

    choices = data.get("choices")
    if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
        preview = (str(data)[:300] + "...") if not isinstance(data, str) else data[:300] + "..."
        raise RuntimeError(f"Unexpected response schema from solver at {url}. Preview: {preview}")

    return choices[0]["message"]["content"]


# --- NEW: SSE (stream=True) chat call with robust parsing ---
def _http_chat_stream_once(
    base_url: str,
    model: str,
    prompt: str,
    api_key: Optional[str] = None,
    gen_cfg: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
) -> str:
    """
    Streams an OpenAI/SGLang-style /chat/completions response and concatenates
    delta['content'] pieces into a single string.

    Notes:
      - Uses text/event-stream (SSE) with server-chunked lines starting with "data: "
      - Stops on a line containing "[DONE]".
    """
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        # IMPORTANT for SSE:
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        # Avoid gzip+keepalive truncation issues on very long generations:
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    if api_key and api_key not in ("EMPTY", "None", None):
        headers["Authorization"] = f"Bearer {api_key}"

    # Build payload
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,            # <<<<<<<< STREAMING
        "n": 1,
    }
    if gen_cfg:
        norm = dict(gen_cfg)
        if "max_new_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_new_tokens")
        for k, v in norm.items():
            if v is not None:
                payload[k] = v
    payload.setdefault("temperature", 0.0)
    payload.setdefault("max_tokens", 1024)

    url = _build_chat_url(base_url)

    # Separate connect/read timeouts; read applies per-chunk
    # (Requests applies the read timeout to individual socket ops,
    # not the whole stream duration.)
    resp = session.post(
        url, headers=headers, json=payload, stream=True, timeout=(60.0, timeout_s)
    )
    resp.raise_for_status()

    out_chunks: List[str] = []

    # Iterate server-sent events line-by-line
    # decode_unicode=True -> yields str lines
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        # SSE frames look like: "data: {...json...}"
        if raw_line.startswith("data:"):
            data_str = raw_line[len("data:"):].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break
            # Some backends can send multiple JSON objects in one "data:" line
            # separated by \n; split defensively.
            for piece in data_str.split("\n"):
                piece = piece.strip()
                if not piece:
                    continue
                try:
                    evt = json.loads(piece)
                except Exception:
                    # If the server accidentally emits non-JSON lines, skip them
                    continue

                # Handle error payloads if any
                if isinstance(evt, dict) and "error" in evt:
                    err = evt["error"]
                    if isinstance(err, dict):
                        msg = err.get("message", str(err))
                    else:
                        msg = str(err)
                    raise RuntimeError(f"Upstream error: {msg}")

                # Standard OpenAI/SGLang delta format:
                # evt["choices"][0]["delta"]["content"] may be present
                try:
                    choices = evt.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        fragment = delta.get("content")
                        if fragment:
                            out_chunks.append(fragment)
                except Exception:
                    # Be permissive; ignore malformed chunks
                    pass

    return "".join(out_chunks)


def _chat_with_retries(
    base_url: str,
    model: str,
    prompt: str,
    api_key: Optional[str],
    gen_cfg: Dict[str, Any],
    max_retries: int,
    base_delay: float,
    timeout_s: float,
) -> Optional[str]:
    last_err = None
    for attempt in range(max_retries):
        try:
            return _http_chat_stream_once(
                base_url=base_url,
                model=model,
                prompt=prompt,
                api_key=api_key,
                gen_cfg=gen_cfg,
                timeout_s=timeout_s,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    print(f"[cm_reward] HTTP chat (streaming) failed after {max_retries} retries: {last_err}")
    return None



# -------------------------------
# Dataset-specific evaluation
# -------------------------------
def _eval_code(
    dataset_name: str,
    tests: Any,
    code: Optional[str],
    use_tci: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    if not code:
        return False, {"error": "no code extracted from solver output"}

    # Prefer TCI for TACo/APPS/CodeContests if requested and available
    if dataset_name in ["taco", "apps", "code_contests"] and use_tci and _HAS_TCI:
        codetool = TogetherCodeTool()
        from rllm.rewards.code_reward import codetool_check_correctness
        return codetool_check_correctness(tests, code, codetool, is_taco_format=True)

    if dataset_name in ["taco", "apps", "code_contests"]:
        lcb_tests = taco_to_lcb_format(tests)
        return lcb_check_correctness_v2(lcb_tests, code, debug=False)

    if dataset_name == "leetcode":
        return leetcode_check_correctness(tests, code)

    if dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
        tests = json.loads(tests) if isinstance(tests, str) else tests
        return lcb_check_correctness_v2(tests, code, debug=False)

    if dataset_name == "kodcode":
        return kodcode_check_correctness(tests, code)

    if dataset_name == "humanevalplus":
        return humanevalplus_check_correctness(tests, code)

    return False, {"error": f"Dataset {dataset_name} not implemented in context assist reward"}


def _extract_code(text: str) -> Optional[str]:
    code = extract_code_from_model(text)
    if code:
        code = clean_code_main_block(code)
    return code


def _pass_fraction(details: Dict[str, Any]) -> Optional[float]:
    """
    Try to extract a fractional pass metric from details if available.
    Falls back to None if not available.
    """
    # Common patterns you may produce in your evaluators:
    for k_total, k_pass in [
        ("n_total", "n_passed"),
        ("total", "passed"),
        ("num_total", "num_passed"),
    ]:
        if isinstance(details, dict) and k_total in details and k_pass in details:
            try:
                total = float(details[k_total])
                passed = float(details[k_pass])
                if total > 0:
                    return passed / total
            except Exception:
                pass
    return None


# -------------------------------
# Core scorer (single sample)
# -------------------------------
def _score_one_sample(
    data_source: str,
    feedback: Optional[str],
    ground_truth: Any,
    extra_info: Dict[str, Any],
    http_cfg: Dict[str, Any],
    use_solver_cot: bool,
    use_marginal_improvement: bool,
    fractional_shaping: bool,
    use_tci: bool,
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Computes reward for a single sample with optional baseline/marginal logic.
    Returns: (reward, is_correct, metadata)
    """
    # Unpack config
    base_url = http_cfg["base_url"]
    model_name = http_cfg["model_name"]
    api_key = http_cfg.get("api_key")
    gen_cfg = http_cfg.get("gen", {"max_tokens": 16384, "temperature": 0.0})
    timeout_s = float(http_cfg.get("timeout_s", 600.0))
    max_retries = int(http_cfg.get("max_retries", 3))
    base_delay = float(http_cfg.get("base_delay", 2.0))

    # Extract problem/attempts
    problem = (
        extra_info.get("problem")
        or extra_info.get("question")
        or extra_info.get("prompt")
        or ""
    )
    prev_attempts = extra_info.get("prev_attempts") or []

    # Try to pull a baseline from prev_attempts (first attempt)
    baseline_passed = None
    baseline_details = None
    baseline_code = None
    baseline_solver_output = None

    if prev_attempts:
        first = prev_attempts[0]
        baseline_passed = first.get("passed", None)
        baseline_details = first.get("results", None)
        baseline_code = first.get("code", None)
        baseline_solver_output = first.get("solver_output", None)

    # If no feedback, compute/return baseline reward
    if not feedback or not str(feedback).strip():
        if baseline_passed is None:
            # Need to compute a baseline by calling solver w/o feedback
            prompt0 = build_solver_prompt(problem, use_solver_cot, None, prev_attempts)
            text0 = _chat_with_retries(
                base_url, model_name, prompt0, api_key, gen_cfg, max_retries, base_delay, timeout_s
            )
            if text0 is None:
                return 0.0, False, {"error": "baseline HTTP call failed"}
            baseline_solver_output = text0
            baseline_code = _extract_code(text0)
            baseline_passed, baseline_details = _eval_code(data_source, ground_truth, baseline_code, use_tci)
        # Reward is just binary baseline
        reward = 1.0 if baseline_passed else 0.0
        return reward, bool(baseline_passed), {
            "initial_passed": baseline_passed,
            "initial_results": baseline_details,
            "initial_solver_code": baseline_code,
            "initial_solver_output": baseline_solver_output,
            "initial_solver_prompt": build_solver_prompt(problem, use_solver_cot, None, prev_attempts),
            "retry_passed": None,
            "retry_results": None,
            "retry_solver_code": None,
            "retry_solver_output": None,
            "retry_solver_prompt": None,
        }

    # With feedback: compute retry
    retry_prompt = build_solver_prompt(problem, use_solver_cot, feedback, prev_attempts)
    retry_text = _chat_with_retries(
        base_url, model_name, retry_prompt, api_key, gen_cfg, max_retries, base_delay, timeout_s
    )
    if retry_text is None:
        return 0.0, False, {"error": "retry HTTP call failed"}

    retry_code = _extract_code(retry_text)
    retry_passed, retry_details = _eval_code(data_source, ground_truth, retry_code, use_tci)
    reward_binary = 1.0 if retry_passed else 0.0

    # fractional shaping if available
    frac_reward = _pass_fraction(retry_details) if fractional_shaping else None
    if frac_reward is not None:
        reward_shaped = frac_reward
    else:
        reward_shaped = reward_binary

    if use_marginal_improvement and baseline_passed is not None:
        # Reward only if we matched or improved from baseline
        if retry_passed and not baseline_passed:
            final_reward = 1.0
        elif retry_passed and baseline_passed:
            final_reward = 1.0
        else:
            final_reward = 0.0
    else:
        final_reward = reward_shaped

    meta = {
        "initial_passed": baseline_passed,
        "initial_results": baseline_details,
        "initial_solver_code": baseline_code,
        "initial_solver_output": baseline_solver_output,
        "initial_solver_prompt": build_solver_prompt(problem, use_solver_cot, None, prev_attempts),
        "retry_passed": retry_passed,
        "retry_results": retry_details,
        "retry_solver_code": retry_code,
        "retry_solver_output": retry_text,
        "retry_solver_prompt": retry_prompt,
    }
    return float(final_reward), bool(retry_passed), meta


# -------------------------------
# PUBLIC: single-sample API (keeps old signature/semantics)
# -------------------------------
def rllm_reward_fn_context_assist(
    data_source: str,
    feedback: str,
    ground_truth: Any,
    problem: str,
    prev_attempts: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> RewardOutput:
    """
    Remote-solver version of your old single-sample entrypoint.
    kwargs:
      - base_url, model_name, api_key
      - gen (temperature, max_tokens / max_new_tokens)
      - timeout_s, max_retries, base_delay
      - use_marginal_improvement (bool)
      - fractional_shaping (bool)
      - use_together_code_interpreter (bool)
    """
    cfg = RewardConfig()
    http_cfg = {
        "base_url": kwargs.get("base_url", kwargs.get("remote_url", "http://localhost:12345/v1")),
        "model_name": kwargs.get("model_name", kwargs.get("solver_model_path", "genrm-demo")),
        "api_key": kwargs.get("api_key", kwargs.get("remote_api_key", "EMPTY")),
        "gen": kwargs.get("gen", {"max_tokens": 16384, "temperature": 0.0}),
        "timeout_s": kwargs.get("timeout_s", 600.0),
        "max_retries": kwargs.get("max_retries", 3),
        "base_delay": kwargs.get("base_delay", 2.0),
    }
    use_solver_cot = bool(kwargs.get("use_solver_cot", False))
    use_marginal = bool(kwargs.get("use_marginal_improvement", True))
    fractional = bool(kwargs.get("fractional_shaping", False))
    use_tci = bool(kwargs.get("use_together_code_interpreter", False))

    reward, is_correct, meta = _score_one_sample(
        data_source=data_source,
        feedback=feedback,
        ground_truth=ground_truth,
        extra_info={"problem": problem, "prev_attempts": prev_attempts or []},
        http_cfg=http_cfg,
        use_solver_cot=use_solver_cot,
        use_marginal_improvement=use_marginal,
        fractional_shaping=fractional,
        use_tci=use_tci,
    )

    # Preserve your old "incorrect_reward" behavior for single-sample if wanted
    if not is_correct and not use_marginal:
        reward = float(cfg.incorrect_reward)

    return RewardOutput(
        reward=float(reward),
        is_correct=bool(is_correct),
        metadata=meta,
    )


# -------------------------------
# PUBLIC: batch API (VERL PPO uses this)
# -------------------------------
def rllm_reward_fn_context_assist_batch(
    data_sources: List[str],
    solution_strs: List[str],      # <-- treat as feedback strings (actions)
    ground_truths: List[Any],
    extra_infos: List[Dict[str, Any]],
    **kwargs,
) -> List[float]:
    """
    Batch scorer compatible with VERL's reward_manager=batch.
    kwargs:
      - base_url, model_name, api_key
      - gen (temperature, max_tokens / max_new_tokens)
      - timeout_s, max_retries, base_delay, max_workers
      - use_marginal_improvement (bool)
      - fractional_shaping (bool)
      - use_together_code_interpreter (bool)
    """
    http_cfg = {
        "base_url": kwargs.get("base_url", kwargs.get("remote_url", "http://localhost:12345/v1")),
        "model_name": kwargs.get("model_name", kwargs.get("solver_model_path", "genrm-demo")),
        "api_key": kwargs.get("api_key", kwargs.get("remote_api_key", "EMPTY")),
        "gen": kwargs.get("gen", {"max_tokens": 16384, "temperature": 0.0}),
        "timeout_s": kwargs.get("timeout_s", 600.0),
        "max_retries": kwargs.get("max_retries", 3),
        "base_delay": kwargs.get("base_delay", 2.0),
    }
    max_workers = int(kwargs.get("max_workers", 32))
    use_solver_cot = bool(kwargs.get("use_solver_cot", False))
    use_marginal = bool(kwargs.get("use_marginal_improvement", True))
    fractional = bool(kwargs.get("fractional_shaping", False))
    use_tci = bool(kwargs.get("use_together_code_interpreter", False))

    results = [0.0] * len(data_sources)

    def _do(i: int) -> float:
        ds = data_sources[i]
        fb = solution_strs[i] if i < len(solution_strs) else ""
        gt = ground_truths[i]
        info = extra_infos[i] if i < len(extra_infos) else {}
        reward, _, _ = _score_one_sample(
            data_source=ds,
            feedback=fb,
            ground_truth=gt,
            extra_info=info,
            http_cfg=http_cfg,
            use_solver_cot=use_solver_cot,
            use_marginal_improvement=use_marginal,
            fractional_shaping=fractional,
            use_tci=use_tci,
        )
        return float(reward)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_do, i): i for i in range(len(data_sources))}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"[cm_reward/batch] sample {i} failed: {e}")
                results[i] = 0.0

    return results
