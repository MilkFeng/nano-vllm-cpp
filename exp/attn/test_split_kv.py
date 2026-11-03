"""GPU contract tests for paged Split-KV decode and caller-owned scratch."""

import ctypes
import math
import unittest

import torch

from baselines import Implementation, custom
from bridge import Bridge, default_library
from cases import Bank, Case, DTYPES
from correctness import compare, references, validate


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SplitKVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = Bridge(default_library())
        if cls.bridge.workspace_size is None:
            raise unittest.SkipTest("rebuild the Split-KV attention library")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def check_case(self, case, dtype_name):
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            bank = Bank(case, DTYPES[dtype_name], 43)
            reference = references(bank)
            implementation = custom(self.bridge, [bank])
            workspace = implementation.launches[0].workspace
            # Every active partition must overwrite scratch; no memset is
            # allowed to be needed between calls, including empty partitions.
            workspace.fill_(math.nan)
            validate(implementation, bank, reference, dtype_name)

    def test_split_boundaries_page_sizes_and_head_mapping(self):
        for dtype_name in DTYPES:
            for heads, kv_heads, dim, page in ((8, 8, 64, 7), (16, 2, 128, 16), (16, 1, 256, 32)):
                for length in (511, 512, 513, 2049, 8193):
                    with self.subTest(dtype=dtype_name, dim=dim, page=page, length=length):
                        self.check_case(Case('split_edges', 'decode', 'test', heads, kv_heads,
                                             dim, (1,), (length,), page_size=page), dtype_name)

    def test_empty_partitions_and_empty_requests(self):
        for dtype_name in DTYPES:
            for causal in (False, True):
                with self.subTest(dtype=dtype_name, causal=causal):
                    self.check_case(Case('ragged_split', 'ragged_decode', 'test', 8, 2, 80,
                                         (1, 0, 1, 1), (1, 37, 513, 8193),
                                         page_size=7, causal=causal), dtype_name)

    def test_partition_maxima_are_merged_stably(self):
        for dtype_name in DTYPES:
            with self.subTest(dtype=dtype_name), torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                case = Case('different_maxima', 'decode', 'test', 16, 2, 128, (1,), (2049,))
                bank = Bank(case, DTYPES[dtype_name], 45)
                bank.query.fill_(1)
                slots = bank.logical_slots[0][:-1]
                # Piecewise scores differ by 120, exercising exponential
                # rescaling across partitions rather than averaging outputs.
                positions = torch.arange(len(slots), device='cuda')
                levels = torch.where(positions < 1024, -60.0, 60.0) / math.sqrt(case.dim)
                bank.key_cache.view(-1, case.hkv, case.dim).index_copy_(
                    0, slots, levels[:, None, None].expand(-1, case.hkv, case.dim).to(bank.dtype))
                bank.new_key.fill_(-60.0 / math.sqrt(case.dim))
                bank.initial_key.copy_(bank.key_cache)
                bank.final_key.copy_(bank.initial_key)
                bank.final_key.view(-1, case.hkv, case.dim).index_copy_(0, bank.append_slots, bank.new_key)
                implementation = custom(self.bridge, [bank])
                validate(implementation, bank, references(bank), dtype_name)

    def test_workspace_size_validation_and_optional_fallback(self):
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            bank = Bank(Case('workspace', 'decode', 'test', 16, 2, 128, (1,), (2049,)), torch.float32, 46)
            reference = references(bank)
            launch = self.bridge.bind(bank)
            self.assertGreater(launch.workspace.numel(), 0)
            launch.arguments.workspace_elements = 1
            with self.assertRaisesRegex(RuntimeError, 'workspace'):
                launch()
            # Existing API callers without scratch must retain correct output.
            launch.arguments.workspace_elements = 0
            launch.arguments.workspace = None
            validate(Implementation('single_cta', [launch]), bank, reference, 'float32')

    def test_capacity_reserved_for_large_batch_serves_smaller_batch(self):
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            bank = Bank(Case('capacity', 'decode', 'test', 16, 2, 128,
                             (1,), (4097,)), torch.float32, 46)
            launch = self.bridge.bind(bank)
            args = launch.arguments
            args.batch, args.tokens = 16, 16
            capacity = ctypes.c_uint64()
            self.assertEqual(self.bridge.workspace_size(ctypes.byref(args), ctypes.byref(capacity)), 0)
            # B=16 itself takes the single-CTA path, but reserving for up to 16
            # requests must still cover the split path used by smaller batches.
            self.assertGreater(capacity.value, 0)
            workspace = torch.full((capacity.value,), math.nan, device='cuda')
            args.batch, args.tokens = 1, 1
            args.workspace, args.workspace_elements = workspace.data_ptr(), workspace.numel()
            validate(Implementation('reserved', [launch]), bank, references(bank), 'float32')

    @torch.inference_mode()
    def test_graph_replay_reuses_poisoned_workspace_on_two_streams(self):
        owners = []
        for seed in (47, 48):
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream):
                bank = Bank(Case('graph', 'decode', 'test', 16, 2, 128,
                                 (1, 1), (17, 4097)), torch.bfloat16, seed)
                reference = references(bank)
                launch = self.bridge.bind(bank)
                launch()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch()
                owners.append((stream, bank, launch, graph, reference))
        for _ in range(3):
            for stream, bank, launch, graph, _ in owners:
                with torch.cuda.stream(stream):
                    bank.reset()
                    launch.workspace.fill_(math.nan)
                    graph.replay()
            for stream, bank, _, _, reference in owners:
                stream.synchronize()
                compare(bank.output, reference[0], 'bfloat16')
                bank.assert_cache_and_guards()


if __name__ == '__main__':
    unittest.main()
