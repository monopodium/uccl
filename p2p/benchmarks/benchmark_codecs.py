"""Single-GPU codec comparison on real LLM weights: LZip-P2P (DietGPU float codec) vs. the lossless
schemes of ZipNN, NeuZip, DFloat11 and ZipCCL, under one setting: the same BF16 tensors, the same GPU.

For each codec it reports the compression ratio (compressed / original bytes) and, where the codec has a
runnable implementation, compression and decompression throughput. Baselines are measured with their own
code where it exists and is runnable on a tensor; otherwise the column is computed and marked as such:

  codec               ratio                    compress            decompress          source
  lzip (DietGPU ANS)  measured                 GPU, measured       GPU, measured       uccl thirdparty dietgpu
  neuzip (nvCOMP ANS  measured                 GPU, measured       GPU, measured       nvidia-nvcomp (NeuZip's codec)
    on exponents)
  zipnn               measured                 CPU, measured       CPU, measured       pip zipnn
  dfloat11            exact Huffman size of    offline CPU (paper) GPU, not measured   computed; DF11 has no
                      the exponent stream                                              per-tensor API
  zipccl              exact size of its fixed  -                   -                   computed; no public code
                      3-bit top-7 exponent code

It also prints a modeled per-tensor transfer time for "encode -> send -> decode" (no overlap, as none of the
baselines overlaps codec work with the network) at --link-gbps, next to the uncompressed transfer time.
The end-to-end LZip-P2P numbers come from benchmark_weight_sync.py / benchmark_uccl_compression.py.

  python benchmarks/benchmark_codecs.py --model /path/to/GLM-4-9B-0414 --link-gbps 400 --out codecs.jsonl
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
import time
from collections import Counter

import torch
from safetensors import safe_open

DEV = torch.device("cuda")


# ----------------------------------------------------------------------------- inputs


def load_bf16_tensors(model_dir: str, min_bytes: int, max_tensors: int) -> list[tuple[str, torch.Tensor]]:
    out = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cuda") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if t.dtype == torch.bfloat16 and t.nbytes >= min_bytes:
                    out.append((name, t.contiguous().flatten()))
                if len(out) >= max_tensors:
                    return out
    assert out, f"no bf16 tensors >= {min_bytes} bytes in {model_dir}"
    return out


def split_bf16(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 -> (exponent bytes, sign|mantissa bytes): the split every exponent coder in this table uses."""
    u = t.view(torch.int16).to(torch.int32) & 0xFFFF
    exp = ((u >> 7) & 0xFF).to(torch.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    return exp, sm


def gpu_time(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


# ----------------------------------------------------------------------------- codecs


def load_dietgpu() -> bool:
    import uccl

    roots = [os.path.dirname(uccl.__file__), os.path.join(os.path.dirname(__file__), "..", "..", "thirdparty", "dietgpu")]
    for root in roots:
        for so in glob.glob(os.path.join(root, "**", "*dietgpu*.so"), recursive=True):
            try:
                torch.ops.load_library(so)
                return True
            except OSError:
                continue
    return False


def bench_dietgpu(t: torch.Tensor, iters: int) -> dict:
    temp = torch.empty(384 << 20, dtype=torch.uint8, device=DEV)
    rows, cols = torch.ops.dietgpu.max_float_compressed_output_size([t])
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=DEV)
    sizes = torch.zeros([1], dtype=torch.int, device=DEV)

    def compress():
        torch.ops.dietgpu.compress_data(True, [t], False, temp, comp, sizes)

    tc = gpu_time(compress, iters)
    nbytes = int(sizes[0].item())
    out = torch.empty_like(t)
    status = torch.empty([1], dtype=torch.uint8, device=DEV)
    words = torch.empty([1], dtype=torch.int32, device=DEV)

    def decompress():
        torch.ops.dietgpu.decompress_data(True, [comp[0][:nbytes]], [out], False, temp, status, words)

    td = gpu_time(decompress, iters)
    assert torch.equal(out.view(torch.int16), t.view(torch.int16)), "dietgpu round trip is not bit-exact"
    return dict(ratio=nbytes / t.nbytes, compress_s=tc, decompress_s=td)


def bench_nvcomp_exponent_ans(t: torch.Tensor, iters: int) -> dict:
    """NeuZip's lossless scheme: nvCOMP ANS on the exponent bytes, sign+mantissa stored raw."""
    from nvidia import nvcomp

    codec = nvcomp.Codec(algorithm="ANS")
    holder = {}

    def compress():
        exp, sm = split_bf16(t)
        holder["enc"] = codec.encode(nvcomp.as_array(exp))
        holder["sm"] = sm

    tc = gpu_time(compress, iters)
    enc = holder["enc"]
    enc_bytes = torch.as_tensor(enc, device=DEV).numel()

    def decompress():
        exp = torch.as_tensor(codec.decode(enc), device=DEV).view(torch.uint8).to(torch.int32)
        sm = holder["sm"].to(torch.int32)
        u = ((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)
        holder["out"] = u.to(torch.int16)

    td = gpu_time(decompress, iters)
    assert torch.equal(holder["out"], t.view(torch.int16)), "nvcomp exponent round trip is not bit-exact"
    return dict(ratio=(enc_bytes + t.numel()) / t.nbytes, compress_s=tc, decompress_s=td)


def bench_zipnn(t: torch.Tensor, iters: int, threads: int) -> dict:
    import zipnn

    z = zipnn.ZipNN(input_format="torch", bytearray_dtype="bfloat16", threads=threads)
    cpu = t.cpu()
    start = time.perf_counter()
    for _ in range(iters):
        blob = z.compress(cpu)
    tc = (time.perf_counter() - start) / iters
    start = time.perf_counter()
    for _ in range(iters):
        back = z.decompress(blob)
    td = (time.perf_counter() - start) / iters
    assert torch.equal(back.view(torch.int16).flatten(), cpu.view(torch.int16)), "zipnn round trip is not bit-exact"
    return dict(ratio=len(blob) / t.nbytes, compress_s=tc, decompress_s=td)


def huffman_bits(counts: Counter) -> int:
    """Total bits of an optimal Huffman code for this symbol histogram (what DFloat11 stores)."""
    if len(counts) == 1:
        return sum(counts.values())
    heap = list(counts.values())
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        total += a + b
        heapq.heappush(heap, a + b)
    return total


def exponent_histogram(t: torch.Tensor) -> Counter:
    exp, _ = split_bf16(t)
    hist = torch.bincount(exp.to(torch.int64), minlength=256).cpu().tolist()
    return Counter({e: c for e, c in enumerate(hist) if c})


def ratio_dfloat11(t: torch.Tensor, hist: Counter) -> dict:
    bits = 8 * t.numel() + huffman_bits(hist)  # raw sign+mantissa byte + Huffman-coded exponent
    return dict(ratio=bits / 8 / t.nbytes)


def ratio_zipccl(t: torch.Tensor, hist: Counter) -> dict:
    # 3-bit code for the 7 most frequent exponents (an oracle top-7, i.e. an upper bound on ZipCCL's
    # analytic sigma-based choice), the 8th code word escapes to a raw 8-bit exponent.
    top7 = sum(c for _, c in hist.most_common(7))
    escapes = t.numel() - top7
    bits = 8 * t.numel() + 3 * t.numel() + 8 * escapes
    return dict(ratio=bits / 8 / t.nbytes)


# ----------------------------------------------------------------------------- driver


def modeled_transfer_s(nbytes: int, row: dict, link_GBps: float) -> float | None:
    if "compress_s" not in row:
        return None
    return row["compress_s"] + nbytes * row["ratio"] / (link_GBps * 1e9) + row["decompress_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-bytes", type=int, default=16 << 20)
    ap.add_argument("--max-tensors", type=int, default=24)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--zipnn-threads", type=int, default=16)
    ap.add_argument("--link-gbps", type=float, default=400.0, help="per-GPU NIC line rate for the modeled column")
    ap.add_argument("--codecs", default="lzip,neuzip,zipnn,dfloat11,zipccl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codecs = args.codecs.split(",")
    if "lzip" in codecs and not load_dietgpu():
        print("[warn] dietgpu torch ops not found; build uccl with USE_DIETGPU=1. Skipping lzip.")
        codecs.remove("lzip")
    link_GBps = args.link_gbps / 8
    totals: dict[str, dict] = {}
    for name, t in load_bf16_tensors(args.model, args.min_bytes, args.max_tensors):
        hist = exponent_histogram(t)
        rows = {}
        for codec in codecs:
            try:
                if codec == "lzip":
                    rows[codec] = bench_dietgpu(t, args.iters)
                elif codec == "neuzip":
                    rows[codec] = bench_nvcomp_exponent_ans(t, args.iters)
                elif codec == "zipnn":
                    rows[codec] = bench_zipnn(t, max(1, args.iters // 5), args.zipnn_threads)
                elif codec == "dfloat11":
                    rows[codec] = ratio_dfloat11(t, hist)
                elif codec == "zipccl":
                    rows[codec] = ratio_zipccl(t, hist)
            except ImportError as e:
                print(f"[warn] {codec} unavailable ({e}); skipping")
                codecs = [c for c in codecs if c != codec]
        raw_s = t.nbytes / (link_GBps * 1e9)
        for codec, row in rows.items():
            row["modeled_transfer_s"] = modeled_transfer_s(t.nbytes, row, link_GBps)
            acc = totals.setdefault(codec, Counter())
            acc["bytes"] += t.nbytes
            acc["comp_bytes"] += row["ratio"] * t.nbytes
            for k in ("compress_s", "decompress_s", "modeled_transfer_s"):
                if row.get(k) is not None:
                    acc[k] += row[k]
        totals.setdefault("uncompressed", Counter())["modeled_transfer_s"] += raw_s
        record = dict(tensor=name, MB=t.nbytes / 2**20, uncompressed_transfer_s=raw_s, **rows)
        print(json.dumps(record))
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")

    print("\n== summary over all measured tensors ==")
    print(f"{'codec':<10} {'ratio':>6} {'comp GB/s':>10} {'decomp GB/s':>12} {'modeled xfer s':>15}")
    total_bytes = totals[codecs[0]]["bytes"] if codecs else 0
    for codec, acc in totals.items():
        ratio = acc["comp_bytes"] / acc["bytes"] if acc.get("bytes") else 1.0
        cgbps = acc["bytes"] / acc["compress_s"] / 1e9 if acc.get("compress_s") else float("nan")
        dgbps = acc["bytes"] / acc["decompress_s"] / 1e9 if acc.get("decompress_s") else float("nan")
        xfer = acc.get("modeled_transfer_s", float("nan"))
        print(f"{codec:<10} {ratio:>6.3f} {cgbps:>10.1f} {dgbps:>12.1f} {xfer:>15.4f}")
    print(f"(total {total_bytes / 1e9:.2f} GB; modeled = compress + compressed/link + decompress, no overlap)")


if __name__ == "__main__":
    main()
