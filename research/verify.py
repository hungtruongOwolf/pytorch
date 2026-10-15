#!/usr/bin/env python
"""Verify a PyTorch 2.3.0 source build. Run by 'bash research/build.sh verify'.

Checks, per GPU: the compiled architectures cover the card, fp32 and fp16 matmul.
On device 0: autograd and the profiler. Across processes: a real all-reduce with the
gloo backend (CPU, always) and with NCCL (one process per GPU, when there are two or
more). Ends with a manifest of the exact versions, to paste into the wiki.

Run it from outside the PyTorch source tree: inside it, `import torch` resolves to
the source folder instead of the built extension.
"""
import os
import platform
import shutil
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

failures = []


def check(name, good, detail=""):
    print("%-16s: %s%s" % (name, "ok" if good else "FAILED", (" " + detail) if detail else ""))
    if not good:
        failures.append(name)


def allreduce_worker(rank, world, init_file, backend):
    """One process of the all-reduce test. Module-level so mp.spawn can import it."""
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method="file://" + init_file, rank=rank, world_size=world)
    device = "cuda" if backend == "nccl" else "cpu"
    t = torch.full((1024,), float(rank + 1), device=device)
    dist.all_reduce(t)
    if device == "cuda":
        torch.cuda.synchronize()
    expect = float(sum(range(1, world + 1)))
    good = bool(torch.all(t == expect).item())
    got = t[0].item()
    dist.barrier()
    dist.destroy_process_group()
    if not good:
        raise SystemExit("rank %d: all-reduce returned %r, expected %r" % (rank, got, expect))
    if rank == 0:
        print("all-reduce %-5s: ok, %d processes, sum %g" % (backend, world, expect))


def run_allreduce(backend, world):
    workdir = tempfile.mkdtemp(prefix="pt230-verify-")
    try:
        mp.spawn(allreduce_worker, args=(world, os.path.join(workdir, "store"), backend),
                 nprocs=world, join=True)
        return True
    except Exception as exc:  # a failing child surfaces here with its traceback
        text = str(exc).strip()
        print("all-reduce %-5s: FAILED: %s" % (backend, text.splitlines()[-1] if text else repr(exc)))
        return False
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def arch_covers(cc):
    """True if the build contains SASS for this card, or PTX old enough to JIT onto it."""
    archs = torch.cuda.get_arch_list()
    if "sm_%d%d" % cc in archs:
        return True
    ptx = [int(a[len("compute_"):]) for a in archs if a.startswith("compute_")]
    return any(p <= cc[0] * 10 + cc[1] for p in ptx)


def gcc_line():
    for line in torch.__config__.show().splitlines():
        if "GCC" in line:
            return line.strip(" -")
    return "unknown"


def nccl_version():
    try:
        return ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        return "not built in"


def first_line(exc):
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else repr(exc)


def main():
    check("torch version", torch.__version__.startswith("2.3.0"), torch.__version__)
    if not torch.cuda.is_available():
        check("CUDA available", False, "see the troubleshooting table in BUILD.md")
        sys.exit(1)

    n = torch.cuda.device_count()
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        check("gpu %d arch" % i, arch_covers(cc),
              "%s sm_%d%d, build has %s" % (name, cc[0], cc[1], " ".join(torch.cuda.get_arch_list())))
        try:
            with torch.cuda.device(i):
                a = torch.randn(2048, 2048, device="cuda")
                b = a @ a
                h = a.half() @ a.half()
                torch.cuda.synchronize()
                good = bool(torch.isfinite(b).all().item()) and bool(torch.isfinite(h.float()).all().item())
            check("gpu %d matmul" % i, good, "fp32 and fp16")
        except Exception as exc:
            check("gpu %d matmul" % i, False, first_line(exc))

    try:
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        (x * x).sum().backward()
        torch.cuda.synchronize()
        check("autograd", x.grad is not None and tuple(x.grad.shape) == (64, 64))
    except Exception as exc:
        check("autograd", False, first_line(exc))

    try:
        from torch.profiler import ProfilerActivity, profile
        a = torch.randn(2048, 2048, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            (a @ a).sum().item()
        check("profiler", len(prof.key_averages()) > 0, "%d entries" % len(prof.key_averages()))
    except Exception as exc:
        check("profiler", False, first_line(exc))

    if dist.is_available():
        if not run_allreduce("gloo", 2):
            failures.append("all-reduce gloo")
    else:
        check("distributed", False, "torch.distributed is not available in this build")
    if n >= 2:
        if dist.is_nccl_available():
            if not run_allreduce("nccl", n):
                failures.append("all-reduce nccl")
        else:
            check("all-reduce nccl", False, "this build has no NCCL (was it built with USE_NCCL=0?)")
    else:
        print("all-reduce nccl : skipped, only one GPU on this host")

    print("\n--- manifest (paste into the wiki) ---")
    print("host        :", socket.gethostname())
    print("python      :", platform.python_version())
    print("torch       :", torch.__version__)
    print("git version :", torch.version.git_version)
    print("CUDA / cuDNN:", torch.version.cuda, "/", torch.backends.cudnn.version())
    print("NCCL        :", nccl_version())
    print("built with  :", gcc_line())
    print("arch list   :", " ".join(torch.cuda.get_arch_list()))
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        print("gpu %d       : %s, sm_%d%d" % (i, torch.cuda.get_device_name(i), cc[0], cc[1]))

    if failures:
        print("\nFAILED:", ", ".join(failures))
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
