"""Whole-model weight-sync benchmark for LZip-P2P (UCCL-P2P + lossless compression).

Emulates the RL weight update of a trainer -> rollout engine: rank 0 (trainer) holds every tensor of a
HuggingFace checkpoint on its GPU and RDMA-WRITEs all of them into identically shaped tensors on rank 1
(rollout engine). One iteration = writing the full model + the completion fence, i.e. the time after
which the rollout engine may run with the new weights. Every iteration is verified bit-exact.

Run it twice per model, once per strategy, and compare `sync_s`:

  # baseline UCCL-P2P
  UCCL_P2P_COMPRESS_STRATEGY=none torchrun --nnodes=2 --nproc_per_node=1 --node-rank=<0|1> \
      --master_addr=<IP> benchmarks/benchmark_weight_sync.py --model /path/to/GLM-4-9B-0414
  # LZip-P2P split-send
  UCCL_P2P_COMPRESS_STRATEGY=split_only UCCL_P2P_COMPRESS_BUFFER_BYTES=$((1<<30)) torchrun ... same

Tensors are written one at a time in checkpoint order (as miles writes parameter groups). A tensor takes
the compressed path when it is a float tensor of at least UCCL_P2P_MIN_COMPRESS_BYTES (default 16 MiB)
and fits the compression buffer; every other tensor is written raw.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.distributed as dist
from safetensors import safe_open

from uccl import p2p

_FLOAT_TYPES = {
    torch.bfloat16: p2p.FloatType.kBFloat16,
    torch.float16: p2p.FloatType.kFloat16,
    torch.float32: p2p.FloatType.kFloat32,
    torch.float8_e4m3fn: p2p.FloatType.kFloat8E4M3FN,
    torch.float8_e5m2: p2p.FloatType.kFloat8E5M2,
}
_MIN_COMPRESS_BYTES = int(os.environ.get("UCCL_P2P_MIN_COMPRESS_BYTES", 16 << 20))
_COMPRESS_BUFFER_BYTES = int(os.environ.get("UCCL_P2P_COMPRESS_BUFFER_BYTES", 2 << 30))
_STRATEGY = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "none").lower()
_COMPRESSING = _STRATEGY in ("split", "split_only")


def _load_checkpoint(model_dir: str, max_tensors: int | None) -> list[tuple[str, torch.Tensor]]:
    tensors = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cuda") as f:
            for name in f.keys():
                tensors.append((name, f.get_tensor(name).contiguous()))
                if max_tensors is not None and len(tensors) >= max_tensors:
                    return tensors
    assert tensors, f"no *.safetensors under {model_dir}"
    return tensors


def _compresses(t: torch.Tensor) -> bool:
    return (
        _COMPRESSING
        and t.dtype in _FLOAT_TYPES
        and _MIN_COMPRESS_BYTES <= t.nbytes <= _COMPRESS_BUFFER_BYTES
        and t.nbytes % 2 == 0
    )


def _send_obj(obj, dst: int) -> None:
    payload = torch.frombuffer(bytearray(json.dumps(obj).encode()), dtype=torch.uint8)
    dist.send(torch.tensor([payload.numel()], dtype=torch.int64), dst=dst)
    dist.send(payload, dst=dst)


def _recv_obj(src: int):
    n = torch.zeros(1, dtype=torch.int64)
    dist.recv(n, src=src)
    payload = torch.zeros(int(n.item()), dtype=torch.uint8)
    dist.recv(payload, src=src)
    return json.loads(bytes(payload.tolist()).decode())


_SIZE_BUCKETS = [(0, 1 << 20, "<1MB"), (1 << 20, 16 << 20, "1-16MB"), (16 << 20, 64 << 20, "16-64MB"),
                 (64 << 20, 256 << 20, "64-256MB"), (256 << 20, 1 << 62, ">=256MB")]


def _size_distribution(tensors: list[torch.Tensor]) -> dict:
    """Tensor count and byte share per size bucket: where the bytes of a model's weight sync are."""
    total = sum(t.nbytes for t in tensors)
    out = {}
    for lo, hi, label in _SIZE_BUCKETS:
        sel = [t for t in tensors if lo <= t.nbytes < hi]
        out[label] = dict(count=len(sel), byte_share=sum(t.nbytes for t in sel) / total,
                          compressed=bool(sel) and all(_compresses(t) for t in sel))
    return out


def _checksum(tensors: list[torch.Tensor]) -> int:
    # order-sensitive, bit-exact: sum of int64 views weighted by position
    acc = torch.zeros((), dtype=torch.int64, device="cuda")
    for i, t in enumerate(tensors):
        b = t.view(torch.uint8).flatten()
        pad = (-b.numel()) % 8
        if pad:
            b = torch.cat([b, torch.zeros(pad, dtype=torch.uint8, device=b.device)])
        acc += b.view(torch.int64).sum() * (i + 1)
    return int(acc.item())


def _run_receiver(args, ep: "p2p.Endpoint", specs: list) -> None:
    dsts = [torch.empty(shape, dtype=getattr(torch, dtype), device="cuda") for _, shape, dtype in specs]
    blobs = []
    for t in dsts:
        ok, mr_id = ep.reg(t.data_ptr(), t.nbytes)
        assert ok
        ok, blob = ep.advertise(mr_id, t.data_ptr(), t.nbytes)
        assert ok
        blobs.append(blob.hex())
    _send_obj(blobs, dst=0)
    for it in range(args.warmup + args.iters):
        for t in dsts:
            t.fill_(0)
        torch.cuda.synchronize()
        _send_obj("ready", dst=0)
        _recv_obj(src=0)  # writer finished, fence included
        torch.cuda.synchronize()
        _send_obj(_checksum(dsts), dst=0)


def _run_sender(args, ep: "p2p.Endpoint", conn_id: int, named: list) -> None:
    blobs = [bytes.fromhex(b) for b in _recv_obj(src=1)]
    plan = []
    for (name, t), blob in zip(named, blobs, strict=True):
        if _compresses(t):
            ok, mr_id = ep.reg_compress_only(t.data_ptr(), t.nbytes, _FLOAT_TYPES[t.dtype])
        else:
            ok, mr_id = ep.reg(t.data_ptr(), t.nbytes)
        assert ok, name
        plan.append((name, t, mr_id, blob))
    expected = _checksum([t for _, t in named])
    total_bytes = sum(t.nbytes for _, t in named)
    compressed_bytes = sum(t.nbytes for _, t in named if _compresses(t))
    times = []
    for it in range(args.warmup + args.iters):
        _recv_obj(src=1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for name, t, mr_id, blob in plan:
            assert ep.write(conn_id, mr_id, t.data_ptr(), t.nbytes, blob), name
        assert ep.flush_compressed_writes(conn_id, timeout_ms=60_000), "fence timed out"
        elapsed = time.perf_counter() - start
        _send_obj("done", dst=1)
        got = _recv_obj(src=1)
        assert got == expected, f"iter {it}: checksum mismatch, transfer is not bit-exact"
        if it >= args.warmup:
            times.append(elapsed)
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1)) ** 0.5
    result = dict(
        model=os.path.basename(os.path.normpath(args.model)),
        strategy=_STRATEGY,
        num_tensors=len(named),
        total_GB=total_bytes / 1e9,
        compressed_path_fraction=compressed_bytes / total_bytes,
        iters=len(times),
        warmup=args.warmup,
        sync_s_median=median,
        sync_s_mean=mean,
        sync_s_std=std,
        sync_s_min=times[0],
        sync_s_max=times[-1],
        size_distribution=_size_distribution([t for _, t in named]),
        effective_GBps=total_bytes / 1e9 / median,
        bit_exact=True,
    )
    print(json.dumps(result))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir with *.safetensors")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tensors", type=int, default=None, help="debug: only the first N tensors")
    ap.add_argument("--out", default=None, help="append the JSON result to this file (rank 0)")
    args = ap.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    local_gpu = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_gpu)
    ep = p2p.Endpoint(local_gpu)

    if rank == 0:
        named = _load_checkpoint(args.model, args.max_tensors)
        _send_obj([(n, list(t.shape), str(t.dtype).removeprefix("torch.")) for n, t in named], dst=1)
        _send_obj(bytes(ep.get_metadata()).hex(), dst=1)
        remote = bytes.fromhex(_recv_obj(src=1))
        ip, port, gpu = p2p.Endpoint.parse_metadata(remote)
        ok, conn_id = ep.connect(ip, gpu, remote_port=port)
        assert ok, "connect failed"
        _run_sender(args, ep, conn_id, named)
    else:
        specs = _recv_obj(src=0)
        _recv_obj(src=0)
        _send_obj(bytes(ep.get_metadata()).hex(), dst=0)
        ok, _, _, _ = ep.accept()
        assert ok, "accept failed"
        _run_receiver(args, ep, specs)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
