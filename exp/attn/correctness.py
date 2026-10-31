"""Independent math references and checks of the full cache mutation contract."""

import math

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from cases import SPEC


def canonical(output, bank):
    if output.ndim == 4:
        output = output.transpose(1, 2).reshape_as(bank.output)
    return output


def compare(actual, expected, dtype_name):
    tolerance = SPEC["correctness"]["tolerances"][dtype_name]
    if actual.numel() == 0:
        return {
            "max_abs_error": 0.0,
            "relative_l2_error": 0.0,
            "max_tolerance_ratio": 0.0,
        }
    difference = (actual.double() - expected.double()).abs()
    ratio = difference / (
        tolerance["atol"] + tolerance["rtol"] * expected.double().abs()
    )
    if not bool(torch.all(torch.isfinite(actual))) or not bool(torch.all(ratio <= 1)):
        raise AssertionError(
            f"attention output mismatch: max abs={difference.max().item():.6g}, "
            f"max tolerance ratio={ratio.max().item():.6g}"
        )
    denominator = expected.double().square().sum().clamp_min(1e-30)
    return {
        "max_abs_error": difference.max().item(),
        "relative_l2_error": (difference.square().sum() / denominator).sqrt().item(),
        "max_tolerance_ratio": ratio.max().item(),
    }


def references(bank):
    """FP32 SDPA-Math with TF32 disabled, plus independent FP64 sampled rows."""
    case = bank.case
    expected = torch.empty_like(bank.output, dtype=torch.float32)
    sample_indices, sample_values = [], []
    query_start = 0
    for batch, (q_length, k_length, prefix) in enumerate(
        zip(case.query_lengths, case.kv_lengths, case.prefixes)
    ):
        if not q_length:
            continue
        slots = bank.logical_slots[batch]
        key = bank.final_key.view(-1, case.hkv, case.dim).index_select(0, slots)
        value = bank.final_value.view(-1, case.hkv, case.dim).index_select(0, slots)
        query = bank.query[query_start : query_start + q_length]
        visible = torch.arange(k_length, device="cuda")[None, :] <= (
            prefix + torch.arange(q_length, device="cuda")[:, None]
        )
        mask = visible if case.causal else None
        with sdpa_kernel([SDPBackend.MATH]):
            result = functional.scaled_dot_product_attention(
                query.float().transpose(0, 1).unsqueeze(0),
                key.float().transpose(0, 1).unsqueeze(0),
                value.float().transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                is_causal=False,
                enable_gqa=True,
                dropout_p=0.0,
            )
        expected[query_start : query_start + q_length] = result.squeeze(0).transpose(
            0, 1
        )
        rows = (
            torch.arange(q_length, device="cuda")
            if q_length <= 32
            else (
                torch.linspace(0, q_length - 1, 16, device="cuda")
                .round()
                .long()
                .unique()
            )
        )
        q64 = query.index_select(0, rows).double().transpose(0, 1)
        k64 = key.double().transpose(0, 1).repeat_interleave(case.hq // case.hkv, dim=0)
        v64 = (
            value.double().transpose(0, 1).repeat_interleave(case.hq // case.hkv, dim=0)
        )
        scores = (q64 @ k64.transpose(-1, -2)) / math.sqrt(case.dim)
        if case.causal:
            scores.masked_fill_(~visible.index_select(0, rows), -math.inf)
        result64 = (scores.softmax(dim=-1) @ v64).transpose(0, 1)
        sample_indices.append(rows + query_start)
        sample_values.append(result64)
        query_start += q_length
    indices = (
        torch.cat(sample_indices)
        if sample_indices
        else torch.empty(0, dtype=torch.long, device="cuda")
    )
    values = torch.cat(sample_values) if sample_values else expected.double()
    compare(expected.index_select(0, indices), values, "float32")
    return expected, indices, values


def validate(implementation, bank, reference, dtype_name):
    expected, indices, expected64 = reference
    bank.reset()
    actual = canonical(implementation.launches[0](), bank).clone()
    torch.cuda.synchronize()
    full = compare(actual, expected, dtype_name)
    full["fp64_sampled_rows"] = len(indices) * bank.case.hq
    full["fp64"] = compare(actual.index_select(0, indices), expected64, dtype_name)
    bank.assert_cache_and_guards()
    repeated = canonical(implementation.launches[0](), bank)
    compare(repeated, actual, dtype_name)
    bank.assert_cache_and_guards()
    return full
