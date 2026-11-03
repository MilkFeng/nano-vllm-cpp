"""Implementations with explicit compute semantics and baseline configuration."""

from dataclasses import dataclass, field
import importlib.metadata
import math
import time
import warnings

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class Implementation:
    name: str
    launches: list
    metadata: dict = field(default_factory=dict)
    correctness: dict = field(default_factory=dict)
    timer: object = None
    owners: list = field(default_factory=list)


class Unsupported(RuntimeError):
    pass


def classify_failure(error):
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "not_installed"
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "memory_limit"
    if isinstance(error, AssertionError):
        return "incorrect"
    text = str(error).lower()
    reasons = (
        "not supported",
        "unsupported",
        "only support",
        "no available kernel",
        "no viable backend",
        "no kernel image",
        "does not support",
        "cannot support",
        "not compiled with",
    )
    return (
        "unsupported"
        if isinstance(error, Unsupported) or any(reason in text for reason in reasons)
        else "error"
    )


def custom(bridge, banks):
    launches = [bridge.bind(bank) for bank in banks]
    return Implementation(
        "custom",
        launches,
        {
            "source": "src/kernels/cuda/attention.cu",
            "append_included": True,
            "output_layout": "NHD",
            "workspace_bytes": sum(launch.workspace.numel() * 4 for launch in launches),
        },
    )


def append_torch(bank):
    case = bank.case
    bank.key_cache.view(-1, case.hkv, case.dim).index_copy_(
        0, bank.append_slots, bank.new_key
    )
    bank.value_cache.view(-1, case.hkv, case.dim).index_copy_(
        0, bank.append_slots, bank.new_value
    )


def sdpa(banks, backend_name):
    backend = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[backend_name]
    launches, owners = [], []
    for bank in banks:
        case = bank.case
        if len(set(case.query_lengths)) != 1 or case.query_lengths != case.kv_lengths:
            raise Unsupported(
                "SDPA performance baseline is limited to uniform contiguous prefill"
            )
        batch = len(case.query_lengths)
        length = case.query_lengths[0]
        query = bank.query.view(batch, length, case.hq, case.dim).transpose(1, 2)
        key, value = bank.dense_cache_views()
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        def launch(q=query, k=key, v=value, item=bank):
            append_torch(item)
            with sdpa_kernel([backend]):
                result = functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True
                )
            item.output.copy_(result.transpose(1, 2).reshape_as(item.output))
            return item.output

        launches.append(launch)
        owners.append((query, key, value))
    return Implementation(
        "sdpa_" + backend_name,
        launches,
        {
            "requested_backend": backend_name,
            "append_included": True,
            "layout_conversion_included": True,
            "output_layout": "NHD",
        },
        owners=owners,
    )


def verify_sdpa_backend(implementation):
    expected = {
        "sdpa_math": "_scaled_dot_product_attention_math",
        "sdpa_flash": "_scaled_dot_product_flash_attention",
        "sdpa_cudnn": "_scaled_dot_product_cudnn_attention",
    }[implementation.name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            output = implementation.launches[0]()
            torch.cuda.synchronize()
    operators = sorted(
        {
            event.key
            for event in trace.key_averages()
            if "scaled_dot_product" in event.key
        }
    )
    if not any(expected in operator for operator in operators):
        raise RuntimeError(f"requested backend was not observed: {operators}")
    implementation.metadata["observed_ops"] = operators
    implementation.metadata["warnings"] = [str(item.message) for item in caught]
    implementation.metadata["output_strides"] = list(output.stride())


def flashinfer(banks, workspace, config):
    import flashinfer as library
    from flashinfer.jit.utils import dtype_map

    if banks[0].dtype not in dtype_map:
        raise Unsupported(
            f"FlashInfer {library.__version__} JIT attention has no "
            f"{banks[0].dtype} entry in its installed dtype map"
        )

    launches, owners, backend_names = [], [], []
    started = time.perf_counter()
    for bank in banks:
        case = bank.case
        if case.phase in ("decode", "ragged_decode"):
            wrapper = library.BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                use_cuda_graph=True,
                use_tensor_cores=config["tensor_cores"],
                backend=config["backend"],
                paged_kv_indptr_buffer=bank.indptr,
                paged_kv_indices_buffer=bank.indices,
                paged_kv_last_page_len_buffer=bank.last_page_len,
            )
            wrapper.plan(
                bank.indptr,
                bank.indices,
                bank.last_page_len,
                case.hq,
                case.hkv,
                case.dim,
                case.page_size,
                q_data_type=bank.dtype,
                kv_data_type=bank.dtype,
                sm_scale=1 / math.sqrt(case.dim),
                pos_encoding_mode="NONE",
            )
        else:
            wrapper = library.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                use_cuda_graph=True,
                backend=config["backend"],
                qo_indptr_buf=bank.cu_query,
                paged_kv_indptr_buf=bank.indptr,
                paged_kv_indices_buf=bank.indices,
                paged_kv_last_page_len_buf=bank.last_page_len,
            )
            wrapper.plan(
                bank.cu_query,
                bank.indptr,
                bank.indices,
                bank.last_page_len,
                case.hq,
                case.hkv,
                case.dim,
                case.page_size,
                causal=case.causal,
                q_data_type=bank.dtype,
                kv_data_type=bank.dtype,
                use_fp16_qk_reduction=False,
                sm_scale=1 / math.sqrt(case.dim),
                pos_encoding_mode="NONE",
            )

        def launch(item=bank, plan=wrapper):
            library.page.append_paged_kv_cache(
                item.new_key,
                item.new_value,
                item.batch_indices,
                item.positions,
                (item.key_cache, item.value_cache),
                item.indices,
                item.indptr,
                item.last_page_len,
                kv_layout="NHD",
            )
            return plan.run(
                item.query, (item.key_cache, item.value_cache), out=item.output
            )

        launches.append(launch)
        owners.append(wrapper)
        backend_names.append(
            getattr(
                wrapper, "resolved_backend", getattr(wrapper, "_backend", "unknown")
            )
        )
    torch.cuda.current_stream().synchronize()
    return Implementation(
        "flashinfer",
        launches,
        {
            "version": importlib.metadata.version("flashinfer-python"),
            "configuration": config,
            "resolved_backends": backend_names,
            "append_included": True,
            "planning_seconds": time.perf_counter() - started,
            "float_workspace_bytes": workspace.numel(),
            "output_layout": "NHD",
        },
        owners=owners,
    )


def flash_attention_optional(banks):
    import flash_attn

    launches = []
    for bank in banks:
        case = bank.case
        batch = len(case.query_lengths)
        length = case.query_lengths[0]
        query = bank.query.view(batch, length, case.hq, case.dim)
        key, value = bank.dense_cache_views()

        def launch(q=query, k=key, v=value, item=bank):
            append_torch(item)
            item.output.copy_(
                flash_attn.flash_attn_func(
                    q, k, v, dropout_p=0.0, causal=True
                ).reshape_as(item.output)
            )
            return item.output

        launches.append(launch)
    return Implementation(
        "flash_attention_optional",
        launches,
        {
            "version": importlib.metadata.version("flash-attn"),
            "append_included": True,
            "layout_conversion_included": True,
            "output_layout": "NHD",
        },
    )
