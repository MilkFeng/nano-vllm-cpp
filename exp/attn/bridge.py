"""Zero-copy ctypes binding to the experimental C ABI."""

import ctypes
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


class Arguments(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "query",
            "key_cache",
            "value_cache",
            "new_key",
            "new_value",
            "cu_query",
            "cu_key",
            "block_table",
            "output",
            "stream",
        )
    ]
    _fields_ += [
        (name, ctypes.c_int32)
        for name in (
            "dtype",
            "batch",
            "tokens",
            "query_heads",
            "kv_heads",
            "head_dim",
            "pages",
            "page_size",
            "table_width",
            "max_query",
            "max_key",
            "causal",
        )
    ]


def default_library():
    candidates = [
        ROOT / "build/libnano-vllm-cpp-attn-exp.so",
        ROOT / "build/windows/Release/nano-vllm-cpp-attn-exp.dll",
        ROOT / "build/Release/nano-vllm-cpp-attn-exp.dll",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


class Bridge:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.library = ctypes.CDLL(str(self.path))
        self.library.nano_attn_forward.argtypes = [ctypes.POINTER(Arguments)]
        self.library.nano_attn_forward.restype = ctypes.c_int
        self.library.nano_attn_error.restype = ctypes.c_char_p
        self.library.nano_attn_build_type.restype = ctypes.c_char_p
        self.build_type = self.library.nano_attn_build_type().decode()

    def bind(self, bank):
        case = bank.case
        tensors = (
            bank.query,
            bank.key_cache,
            bank.value_cache,
            bank.new_key,
            bank.new_value,
            bank.cu_query,
            bank.cu_key,
            bank.block_table,
            bank.output,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("bridge requires contiguous CUDA tensors")
        args = Arguments(
            *[tensor.data_ptr() for tensor in tensors],
            torch.cuda.current_stream().cuda_stream,
            {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[bank.dtype],
            len(case.query_lengths),
            sum(case.query_lengths),
            case.hq,
            case.hkv,
            case.dim,
            bank.key_cache.shape[0],
            case.page_size,
            bank.block_table.shape[1],
            max(case.query_lengths, default=0),
            max(case.kv_lengths, default=0),
            case.causal,
        )

        def launch():
            if self.library.nano_attn_forward(ctypes.byref(args)):
                raise RuntimeError(self.library.nano_attn_error().decode())
            return bank.output

        return launch
