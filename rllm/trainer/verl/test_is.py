#!/usr/bin/env python3
# test_is.py
import math
import types
import numpy as np
import torch

from verl.protocol import DataProto
from rllm.trainer.verl.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.ray_trainer import compute_response_mask

# Import your class (adjust path if needed)
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


def _aeq(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol, rtol=0)


def test_per_step_sequence_level():
    """
    Mirrors AgentPPOTrainer's per_step branch:
    - Compute sequence-level IS via veRL helper
    - Multiply tokenwise into advantages on response tokens
    - Check exact expected values
    """
    B, T = 2, 6
    # Per-turn response masks
    response_mask = torch.zeros(B, T, dtype=torch.float32)
    response_mask[0, 2] = 1
    response_mask[0, 3] = 1
    response_mask[1, 4] = 1

    # Create log-ratio (old - rollout) so that:
    # row0 => two tokens with ln 2 each => sequence weight = exp(ln2+ln2)=4
    # row1 => one token with ln 0.5 => sequence weight = 0.5
    log_ratio = torch.zeros(B, T)
    log_ratio[0, 2] = math.log(2.0)
    log_ratio[0, 3] = math.log(2.0)
    log_ratio[1, 4] = math.log(0.5)

    old_lp = log_ratio.clone()
    ro_lp  = torch.zeros_like(old_lp)

    # Build a DataProto like AgentPPOTrainer uses
    tensor_batch = {
        "old_log_probs": old_lp,
        "rollout_log_probs": ro_lp,
        "response_mask": response_mask,
        # Dummy advantages (ones on response tokens) to see pure reweighting
        "advantages": response_mask.clone(),
    }
    batch = DataProto.from_dict(tensors=tensor_batch)

    # === What AgentPPOTrainer does in per_step mode ===
    weights_proto, _ = compute_rollout_importance_weights(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is_level="sequence",
        rollout_is_mode="truncate",
        rollout_is_threshold=10.0,
        rollout_is_threshold_lower=None,  # reciprocal default inside veRL
        rollout_is_veto_threshold=None,   # disable veto for clarity
    )
    w_tok = weights_proto.batch["rollout_is_weights"]  # [B,T]

    batch.batch["advantages"] = batch.batch["advantages"] * w_tok

    # Expected tokenwise
    expected_adv = torch.zeros_like(w_tok)
    expected_adv[0, 2] = 4.0  # 1 * 4
    expected_adv[0, 3] = 4.0
    expected_adv[1, 4] = 0.5  # 1 * 0.5

    assert _aeq(batch.batch["advantages"], expected_adv), (
        f"per_step IS reweighting mismatch:\nGot:\n{batch.batch['advantages']}\nExp:\n{expected_adv}"
    )
    print("[OK] per_step: tokenwise IS weights applied to advantages as expected.")


def test_broadcast_mode_agentppotrainer():
    """
    Mirrors AgentPPOTrainer's broadcast branch:
    - Compute IS on last-step rows
    - Multiply last-step advantages by tokenwise weights
    - Collapse to per-episode IS scalars
    - Call AgentPPOTrainer._stepwise_advantage_broadcast to propagate to earlier steps
    - Verify earlier steps got the correct scalar reweighting
    """
    T = 6

    # ---------- Build LAST-STEP batch (two episodes: idx 0 and 1) ----------
    B_last = 2
    last_resp_mask = torch.zeros(B_last, T, dtype=torch.float32)
    # ep0: two resp tokens
    last_resp_mask[0, 2] = 1
    last_resp_mask[0, 3] = 1
    # ep1: two resp tokens
    last_resp_mask[1, 1] = 1
    last_resp_mask[1, 2] = 1

    # Desired per-episode sequence weights (IS):
    w0, w1 = 3.0, 0.25

    # Set log-ratios so exp(sum) equals the desired seq weight
    log_ratio = torch.zeros(B_last, T)
    log_ratio[0, 2] = math.log(w0) / 2.0
    log_ratio[0, 3] = math.log(w0) / 2.0
    log_ratio[1, 1] = math.log(w1) / 2.0
    log_ratio[1, 2] = math.log(w1) / 2.0

    old_lp_last = log_ratio.clone()
    ro_lp_last  = torch.zeros_like(old_lp_last)

    # Last-step "advantages": ones over response tokens
    last_adv = last_resp_mask.clone()

    last_step_batch = DataProto.from_dict(
        tensors={
            "old_log_probs": old_lp_last,
            "rollout_log_probs": ro_lp_last,
            "response_mask": last_resp_mask,
            "advantages": last_adv.clone(),
        },
        non_tensors={
            "idxs": np.array([0, 1], dtype=np.int64),
            "step_nums": np.array([3, 2], dtype=np.int64),
        },
    )

    # Compute veRL IS on last steps and apply to last-step advantages (as in AgentPPOTrainer)
    weights_proto, _ = compute_rollout_importance_weights(
        old_log_prob=last_step_batch.batch["old_log_probs"],
        rollout_log_prob=last_step_batch.batch["rollout_log_probs"],
        response_mask=last_step_batch.batch["response_mask"],
        rollout_is_level="sequence",
        rollout_is_mode="truncate",
        rollout_is_threshold=10.0,
        rollout_is_threshold_lower=None,
        rollout_is_veto_threshold=None,
    )
    w_tok_last = weights_proto.batch["rollout_is_weights"]
    last_step_batch.batch["advantages"] = last_step_batch.batch["advantages"] * w_tok_last

    # Derive scalar per episode (mean over masked response tokens)
    denom = last_resp_mask.sum(-1).clamp_min(1)

    # ---------- Build OTHER (earlier) STEPS sharing the same episode idxs ----------
    B_other = 3  # two earlier steps for ep0, one for ep1
    other_idxs = np.array([0, 0, 1], dtype=np.int64)
    other_resp_mask = torch.zeros(B_other, T, dtype=torch.float32)
    # ep0 earlier steps
    other_resp_mask[0, 0] = 1
    other_resp_mask[0, 1] = 1
    other_resp_mask[1, 4] = 1
    # ep1 earlier step
    other_resp_mask[2, 5] = 1

    other_step_batch = DataProto.from_dict(
        tensors={"response_mask": other_resp_mask.clone()},
        non_tensors={"idxs": other_idxs},
    )

    # ---------- Call the actual AgentPPOTrainer broadcaster ----------
    # We don't want to construct the whole trainer; call the unbound method
    # with a tiny fake 'self' that supplies just the needed config flag.
    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(
            rllm=types.SimpleNamespace(
                stepwise_advantage=types.SimpleNamespace(normalize_by_steps=False)
            )
        )
    )

    # Ensure masks computed if missing (mirrors trainer)
    if "response_mask" not in other_step_batch.batch:
        other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
    if "response_mask" not in last_step_batch.batch:
        last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)

    # Invoke the class method as unbound with our fake self
    AgentPPOTrainer._stepwise_advantage_broadcast(fake_self, last_step_batch, other_step_batch)

    adv_out = other_step_batch.batch["advantages"]
    # Expected: every response token in ep0 earlier rows == w0, ep1 earlier row == w1
    expected = torch.zeros_like(adv_out)
    expected[0] = other_resp_mask[0] * w0
    expected[1] = other_resp_mask[1] * w0
    expected[2] = other_resp_mask[2] * w1

    assert _aeq(adv_out, expected), f"broadcast IS mismatch:\nGot:\n{adv_out}\nExp:\n{expected}"
    print("[OK] broadcast: AgentPPOTrainer._stepwise_advantage_broadcast applied IS scalars correctly.")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    test_per_step_sequence_level()
    test_broadcast_mode_agentppotrainer()
    print("All AgentPPOTrainer IS tests passed.")


if __name__ == "__main__":
    main()
