"""CUDA Graph sampling; every sample covers a whole number of bank rotations."""

import math
import random
import statistics

import torch


class GraphTimer:
    def __init__(self, launches, calls):
        self.calls = calls
        self.graph = torch.cuda.CUDAGraph()
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.graph(self.graph, stream=torch.cuda.current_stream()):
            for index in range(calls):
                self.output = launches[index % len(launches)]()
        self.graph.replay()
        torch.cuda.current_stream().synchronize()

    def sample(self):
        self.start.record()
        self.graph.replay()
        self.stop.record()
        self.stop.synchronize()
        return self.start.elapsed_time(self.stop) * 1000 / self.calls


def prepare_timer(launches, warmup, max_calls, target_ms):
    for index in range(max(warmup, len(launches))):
        launches[index % len(launches)]()
    torch.cuda.current_stream().synchronize()
    pilot = GraphTimer(launches, len(launches))
    per_call = pilot.sample()
    del pilot
    wanted = min(max_calls, math.ceil(target_ms * 1000 / max(0.1, per_call)))
    calls = max(len(launches), wanted // len(launches) * len(launches))
    return GraphTimer(launches, calls)


def describe(samples, calls):
    ordered = sorted(samples)
    mean = statistics.mean(samples)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "cv": statistics.pstdev(samples) / mean,
        "p10_us": ordered[math.floor((len(samples) - 1) * 0.1)],
        "p90_us": ordered[math.ceil((len(samples) - 1) * 0.9)],
        "calls_per_sample": calls,
    }


def measure_interleaved(implementations, samples, seed):
    generator = random.Random(seed)
    order = list(implementations)
    all_samples = {name: [] for name in order}
    orders = []
    for _ in range(samples):
        generator.shuffle(order)
        orders.append(list(order))
        for name in order:
            all_samples[name].append(implementations[name].timer.sample())
    return {
        name: describe(values, implementations[name].timer.calls)
        for name, values in all_samples.items()
    }, orders
