"""Predeclared shapes, physical page assignments, and mutable cache fixtures."""

from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path

import torch

SPEC = json.loads((Path(__file__).with_name("suite.json")).read_text())
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Case:
    name: str
    phase: str
    profile: str
    hq: int
    hkv: int
    dim: int
    query_lengths: tuple
    kv_lengths: tuple
    page_size: int = 16
    causal: bool = True

    def __post_init__(self):
        if (
            len(self.query_lengths) != len(self.kv_lengths)
            or any(q < 0 or k < q for q, k in zip(self.query_lengths, self.kv_lengths))
            or self.hq < 1
            or self.hkv < 1
            or self.hq % self.hkv
            or not 0 < self.dim <= 256
            or self.page_size < 1
        ):
            raise ValueError("invalid attention case")

    @property
    def prefixes(self):
        return tuple(k - q for q, k in zip(self.query_lengths, self.kv_lengths))

    def document(self):
        return {**asdict(self), "prefix_lengths": self.prefixes}


def expand_cases(suite="full"):
    cases = []
    for profile, head in SPEC["profiles"].items():
        dims = (head["query_heads"], head["kv_heads"], head["head_dim"])
        if suite == "quick":
            for phase, q, k in (
                ("prefill", 128, 128),
                ("decode", 1, 512),
                ("chunked_prefill", 3, 20),
            ):
                cases.append(
                    Case(f"{profile}_{phase}_quick", phase, profile, *dims, (q,), (k,))
                )
            continue
        for grid in SPEC["full"]["grids"]:
            lengths = grid.get("kv_lengths", grid.get("prefix_lengths", [0]))
            for batch, query, length in itertools.product(
                grid["batch_sizes"], grid["query_lengths"], lengths
            ):
                key = length if "kv_lengths" in grid else length + query
                name = f"{profile}_{grid['name']}_b{batch}_q{query}_k{key}"
                cases.append(
                    Case(
                        name,
                        grid["name"],
                        profile,
                        *dims,
                        (query,) * batch,
                        (key,) * batch,
                    )
                )
        for ragged in SPEC["full"]["ragged"]:
            queries = tuple(ragged["query_lengths"])
            keys = tuple(ragged.get("kv_lengths", ()))
            if not keys:
                keys = tuple(q + p for q, p in zip(queries, ragged["prefix_lengths"]))
            cases.append(
                Case(
                    f"{profile}_{ragged['name']}",
                    ragged["name"],
                    profile,
                    *dims,
                    queries,
                    keys,
                )
            )
    return cases


class Bank:
    """Independent addresses with caller-owned state, guards, and metadata."""

    tensor_fields = (
        "query",
        "key_cache",
        "value_cache",
        "new_key",
        "new_value",
        "cu_query",
        "cu_key",
        "block_table",
        "indptr",
        "indices",
        "last_page_len",
        "batch_indices",
        "positions",
        "append_slots",
        "raw_output",
    )

    def __init__(self, case, dtype, seed, identity=False, deterministic=False):
        self.case = case
        self.dtype = dtype
        self.identity = identity
        generator = torch.Generator(device="cpu").manual_seed(seed)
        page_counts = [
            (k + case.page_size - 1) // case.page_size for k in case.kv_lengths
        ]
        used_pages = sum(page_counts)
        width = max(page_counts, default=0)
        order = (
            list(range(used_pages))
            if identity
            else torch.randperm(used_pages, generator=generator).tolist()
        )
        table = torch.full((len(page_counts), width), -1, dtype=torch.int32)
        tables = []
        offset = 0
        for batch, count in enumerate(page_counts):
            ids = order[offset : offset + count]
            tables.append(ids)
            table[batch, :count] = torch.tensor(ids, dtype=torch.int32)
            offset += count
        self.host_tables = tables
        self.block_table = table.cuda()
        self.indices = torch.tensor(order, dtype=torch.int32, device="cuda")
        self.indptr = torch.tensor(
            [0, *itertools.accumulate(page_counts)], dtype=torch.int32, device="cuda"
        )
        self.last_page_len = torch.tensor(
            [(k - 1) % case.page_size + 1 if k else 0 for k in case.kv_lengths],
            dtype=torch.int32,
            device="cuda",
        )
        self.cu_query = torch.tensor(
            [0, *itertools.accumulate(case.query_lengths)],
            dtype=torch.int32,
            device="cuda",
        )
        self.cu_key = torch.tensor(
            [0, *itertools.accumulate(case.kv_lengths)],
            dtype=torch.int32,
            device="cuda",
        )
        batch_ids, positions, slots = [], [], []
        for batch, (query, prefix) in enumerate(zip(case.query_lengths, case.prefixes)):
            for position in range(prefix, prefix + query):
                batch_ids.append(batch)
                positions.append(position)
                slots.append(
                    tables[batch][position // case.page_size] * case.page_size
                    + position % case.page_size
                )
        self.batch_indices = torch.tensor(batch_ids, dtype=torch.int32, device="cuda")
        self.positions = torch.tensor(positions, dtype=torch.int32, device="cuda")
        self.append_slots = torch.tensor(slots, dtype=torch.int64, device="cuda")
        self.logical_slots = []
        for ids, length in zip(tables, case.kv_lengths):
            self.logical_slots.append(
                torch.tensor(
                    [
                        ids[i // case.page_size] * case.page_size + i % case.page_size
                        for i in range(length)
                    ],
                    dtype=torch.int64,
                    device="cuda",
                )
            )

        torch.cuda.manual_seed(seed)
        tokens = sum(case.query_lengths)
        self.query = (torch.randn(tokens, case.hq, case.dim, device="cuda") * 0.5).to(
            dtype
        )
        self.new_key = (
            torch.randn(tokens, case.hkv, case.dim, device="cuda") * 0.5
        ).to(dtype)
        self.new_value = torch.randn_like(self.new_key) * 0.5
        shape = (used_pages + 2, case.page_size, case.hkv, case.dim)
        self.key_cache = (torch.randn(shape, device="cuda") * 0.5).to(dtype)
        self.value_cache = (torch.randn(shape, device="cuda") * 0.5).to(dtype)
        self.key_cache[used_pages:].fill_(-23)
        self.value_cache[used_pages:].fill_(29)
        for ids, length in zip(tables, case.kv_lengths):
            if ids and length % case.page_size:
                self.key_cache[ids[-1], length % case.page_size :].fill_(17)
                self.value_cache[ids[-1], length % case.page_size :].fill_(19)
        if deterministic:
            self.query.zero_()
            self.key_cache.zero_()
            self.new_key.zero_()
            head_offsets = (
                torch.arange(case.hkv, device="cuda", dtype=torch.float32) * 64
            )
            for batch, logical in enumerate(self.logical_slots):
                values = torch.arange(len(logical), device="cuda", dtype=torch.float32)
                self.value_cache.view(-1, case.hkv, case.dim).index_copy_(
                    0,
                    logical,
                    (values[:, None, None] + head_offsets[None, :, None])
                    .expand(-1, case.hkv, case.dim)
                    .to(dtype),
                )
            for i, position in enumerate(positions):
                self.new_value[i].copy_(
                    (position + head_offsets[:, None])
                    .expand(case.hkv, case.dim)
                    .to(dtype)
                )
        self.key_cache.view(-1, case.hkv, case.dim).index_fill_(
            0, self.append_slots, 13
        )
        self.value_cache.view(-1, case.hkv, case.dim).index_fill_(
            0, self.append_slots, -17
        )
        self.initial_key = self.key_cache.clone()
        self.initial_value = self.value_cache.clone()
        self.final_key = self.key_cache.clone()
        self.final_value = self.value_cache.clone()
        self.final_key.view(-1, case.hkv, case.dim).index_copy_(
            0, self.append_slots, self.new_key
        )
        self.final_value.view(-1, case.hkv, case.dim).index_copy_(
            0, self.append_slots, self.new_value
        )
        self.raw_output = torch.full(
            (tokens * case.hq * case.dim + 64,), -31, dtype=dtype, device="cuda"
        )
        self.output = self.raw_output[32:-32].view(tokens, case.hq, case.dim)

    def reset(self):
        self.key_cache.copy_(self.initial_key)
        self.value_cache.copy_(self.initial_value)
        self.raw_output.fill_(-31)

    def clone_addresses(self):
        other = object.__new__(Bank)
        other.__dict__ = self.__dict__.copy()
        for name in self.tensor_fields:
            setattr(other, name, getattr(self, name).clone())
        other.output = other.raw_output[32:-32].view_as(self.output)
        return other

    def bytes(self):
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in self.tensor_fields
        )

    def dense_cache_views(self):
        if not self.identity or len(set(self.case.kv_lengths)) != 1:
            raise ValueError(
                "dense views require uniform lengths and an identity page table"
            )
        batch = len(self.case.kv_lengths)
        length = self.case.kv_lengths[0]
        padded = (
            (length + self.case.page_size - 1) // self.case.page_size
        ) * self.case.page_size
        elements = batch * padded * self.case.hkv * self.case.dim
        key = self.key_cache.flatten()[:elements].view(
            batch, padded, self.case.hkv, self.case.dim
        )[:, :length]
        value = self.value_cache.flatten()[:elements].view(
            batch, padded, self.case.hkv, self.case.dim
        )[:, :length]
        return key, value

    def assert_cache_and_guards(self):
        if not torch.equal(self.key_cache, self.final_key) or not torch.equal(
            self.value_cache, self.final_value
        ):
            raise AssertionError(
                "cache update, old prefix, padding, or unused pages do not match"
            )
        if not bool(torch.all(self.raw_output[:32] == -31)) or not bool(
            torch.all(self.raw_output[-32:] == -31)
        ):
            raise AssertionError("output guard overwritten")

    def metadata(self):
        return {
            "query_strides": list(self.query.stride()),
            "cache_strides": list(self.key_cache.stride()),
            "output_strides": list(self.output.stride()),
            "physical_pages": self.host_tables,
            "bytes_per_bank": self.bytes(),
        }
