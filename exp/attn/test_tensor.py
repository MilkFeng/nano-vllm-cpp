"""Production tensor-core attention: ragged tiles, partitions, and graph reuse."""

import math
import unittest

import torch

from baselines import Implementation, custom
from bridge import Bridge, default_library
from cases import Bank, Case, DTYPES
from correctness import compare, references, validate


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TensorAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = Bridge(default_library())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def test_ragged_tiles_pages_and_empty_partitions(self):
        for dtype in ("float16", "bfloat16"):
            for dim, kv_heads, page in ((128, 8, 7), (256, 2, 16)):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, dim=dim, causal=causal):
                        case = Case("tensor_ragged", "chunked_prefill", "test", 16,
                                    kv_heads, dim, (0, 1, 3, 17, 65),
                                    (0, 1, 130, 530, 2114), page, causal)
                        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                            bank = Bank(case, DTYPES[dtype], 51)
                            reference = references(bank)
                            impl = custom(self.bridge, [bank])
                            impl.launches[0].workspace.fill_(math.nan)
                            validate(impl, bank, reference, dtype)

    def test_split_and_no_workspace_paths(self):
        for dtype in ("float16", "bfloat16"):
            for dim in (128, 256):
                with self.subTest(dtype=dtype, dim=dim):
                    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                        case = Case("tensor_split", "chunked_prefill", "test", 16,
                                    2, dim, (17,), (2066,), page_size=7)
                        bank = Bank(case, DTYPES[dtype], 52)
                        reference = references(bank)
                        launch = self.bridge.bind(bank)
                        launch.workspace.fill_(math.nan)
                        validate(Implementation("split", [launch]), bank, reference, dtype)
                        launch.arguments.workspace_elements = 1
                        with self.assertRaisesRegex(RuntimeError, "workspace"):
                            launch()
                        launch.arguments.workspace = None
                        launch.arguments.workspace_elements = 0
                        validate(Implementation("no_workspace", [launch]), bank, reference, dtype)

    def test_partition_maxima_remain_stable_with_tensor_cores(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype), torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                case = Case("tensor_extreme", "chunked_prefill", "test", 16, 2,
                            256, (17,), (2066,), page_size=7)
                bank = Bank(case, DTYPES[dtype], 55)
                bank.query.fill_(1)
                slots = bank.logical_slots[0][:-17]
                levels = torch.where(torch.arange(len(slots), device="cuda") < 1024,
                                     -60.0, 60.0) / math.sqrt(case.dim)
                bank.key_cache.view(-1, case.hkv, case.dim).index_copy_(
                    0, slots, levels[:, None, None].expand(-1, case.hkv, case.dim).to(bank.dtype))
                bank.new_key.fill_(-60.0 / math.sqrt(case.dim))
                bank.initial_key.copy_(bank.key_cache)
                bank.final_key.copy_(bank.initial_key)
                bank.final_key.view(-1, case.hkv, case.dim).index_copy_(
                    0, bank.append_slots, bank.new_key)
                validate(custom(self.bridge, [bank]), bank, references(bank), dtype)

    @torch.inference_mode()
    def test_two_stream_graphs_overwrite_poisoned_scratch(self):
        owners = []
        for seed, dim in ((53, 128), (54, 256)):
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream):
                case = Case("tensor_graph", "chunked_prefill", "test", 16, 2,
                            dim, (3, 65), (3, 2114), page_size=7)
                bank = Bank(case, torch.bfloat16, seed)
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
                compare(bank.output, reference[0], "bfloat16")
                bank.assert_cache_and_guards()


if __name__ == "__main__":
    unittest.main()
