# Building PyTorch v2.3.0 from Source

This repository — `hungtruongOwolf/pytorch`, branch `research/base-v2.3.0` — is a
fork of PyTorch pinned to the v2.3.0 tag, so that its C++/CUDA internals can be
modified. The compiled library is machine-specific and is never committed, so
every machine has to build it once. `research/build.sh` does that, detecting the
GPU architecture, CUDA version, host compiler and job count from the host it runs
on — nothing here needs editing for a different GPU.

## Requirements

| | |
|---|---|
| OS | Linux (Ubuntu 22.04 LTS verified; other distros need their own package commands) |
| GPU | NVIDIA, with the proprietary driver loaded — `nvidia-smi` must work |
| Driver | >= 530.30.02 for CUDA 12.1, or >= 520.61.05 for CUDA 11.8 |
| CUDA Toolkit | 11.8 or 12.1 (the versions PyTorch 2.3.0 officially targets) |
| cuDNN | 8.9.x matching the CUDA version |
| Python | 3.8-3.11, in a dedicated conda environment |
| RAM | 32 GB comfortable; 16 GB workable but limits `MAX_JOBS` to about 5; 8 GB only with generous swap |
| Disk | 50 GB minimum, 100 GB comfortable, on **ext4** — not NTFS |

`build.sh check` verifies all of this and refuses to build until it is satisfied.

### On WSL2

WSL2 builds and runs CUDA correctly, but differs from native Linux in five ways.
`build.sh` detects WSL and warns; the three setup steps below are manual:

1. **Never install an NVIDIA driver inside WSL.** The GPU is borrowed from the
   Windows driver — that is why `nvidia-smi` works with no Linux driver present.
   Installing one breaks the passthrough. Use NVIDIA's WSL-specific CUDA
   repository, which ships no driver packages:
   `…/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb`, then
   `sudo apt install -y cuda-toolkit-12-1`. Never the `cuda` metapackage, which
   would pull a driver in.
2. **cuDNN is not in the `wsl-ubuntu` repository**, so `libcudnn8` cannot be
   installed with apt. Use NVIDIA's public redistributable instead, which needs
   no account, and unpack it next to CUDA so cmake finds it:
   ```bash
   wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   sudo cp -P cudnn-*/include/cudnn*.h /usr/local/cuda-12.1/include/
   sudo cp -P cudnn-*/lib/libcudnn*    /usr/local/cuda-12.1/lib64/
   sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
   sudo ldconfig
   ```
   `cp -P` is required: it preserves the `libcudnn.so → .so.8 → .so.8.9.7`
   symlink chain, which a plain copy would flatten and break.
3. **WSL gets half the host's RAM by default**, which starves `MAX_JOBS`. Raise it
   in `C:\Users\<name>\.wslconfig` on the Windows side, then `wsl --shutdown`:
   ```ini
   [wsl2]
   memory=12GB
   swap=24GB
   processors=16
   ```
   The large swap matters more than the memory: it is what keeps a peak link step
   from being OOM-killed an hour into a build.

4. **The repository must live in the Linux filesystem** (`~/pytorch`). Building
   from `/mnt/c` or `/mnt/e` is 10-50x slower through the translation layer, and
   NTFS is case-insensitive, which breaks the build outright.
5. **Kernel profiling is restricted.** CUPTI and Nsight hardware counters are
   limited under WSL2. This does not affect building — `torch.profiler` still
   returns entries — but measuring a modified kernel properly eventually needs
   native Linux.

## Quick start

```bash
# 1. Source
git clone https://github.com/hungtruongOwolf/pytorch.git && cd pytorch
git remote add upstream https://github.com/pytorch/pytorch.git
git fetch upstream --tags
git checkout -b research/base-v2.3.0 v2.3.0
git submodule sync --recursive
git submodule update --init --recursive      # after the checkout, not before

# 2. Environment
# conda-forge only: the Anaconda default channels now require accepting a
# commercial Terms of Service, which university use should not need to touch.
conda config --add channels conda-forge && conda config --set channel_priority strict
conda create -n pt230 python=3.10 -y --override-channels -c conda-forge
conda activate pt230
pip install -r requirements.txt
pip install "numpy<2" "setuptools==69.5.1"   # see Version pins below
conda install -y --override-channels -c conda-forge "cmake<4" ninja
conda install -y --override-channels -c conda-forge mkl-static mkl-include
conda install -y -c pytorch magma-cuda121    # optional; match the CUDA version
sudo apt install -y build-essential git wget tmux ccache && ccache -M 25G

# 3. Build
bash research/build.sh check                 # inspect the host, change nothing

tmux new -s ptbuild                          # survive a dropped terminal
conda activate pt230                         # tmux opens a fresh shell: re-activate
bash research/build.sh                       # shows the plan, asks, then builds
bash research/build.sh verify
```

`conda activate` inside `tmux` is not optional — a new shell has no environment
active, and `build.sh` stops with `[FAIL] No conda environment active`.

The first build takes 1-4 hours on a laptop CPU and under an hour on a
workstation. `build.sh` prints its plan and waits for confirmation first, so a
misdetected setting costs seconds rather than hours.

## What build.sh detects

| Setting | Source |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | `nvidia-smi --query-gpu=compute_cap`, deduplicated and sorted |
| `MAX_JOBS` | `min(nproc, RAM_GB / 2)` — PyTorch needs ~2 GB of RAM per compile job, and link steps peak higher |
| `CUDA_HOME` | explicit value, else `nvcc` on `PATH`, else the newest `/usr/local/cuda-*` |
| `CC`, `CXX`, `CUDAHOSTCXX` | the newest installed `gcc-N` that the detected CUDA accepts |
| `USE_NCCL` | `1` when the host has more than one GPU |
| `USE_CUDA`, `USE_CUDNN` | `1` when a toolkit is found |

Compiling device code only for the architectures present is the largest available
build-time saving. Left unset, `nvcc` compiles every architecture PyTorch
supports — several times slower, with nothing usable to show for it. Conversely,
a hardcoded architecture produces a build that succeeds and then fails at run
time with `no kernel image is available for execution on the device`, which is
why this is detected rather than written down.

On a cluster that provides CUDA through environment modules, `module load
cuda/12.1` is enough: `build.sh` finds it on `PATH`.

`MAX_JOBS` is a ceiling, not a target to beat. The 2 GB-per-job estimate holds
for ordinary sources, but the FlashAttention kernels under
`aten/src/ATen/native/transformers/cuda/flash_attn/` take **3-6 GB each** — they
are CUTLASS templates, and nvcc has to expand the whole instantiation tree in
memory. On a 16 GB host, raising `MAX_JOBS` to 8 puts the machine into swap
thrashing during that stretch and makes the build *slower*, not faster. See
the troubleshooting entry below for how to tell thrashing from slow compiling.

## Overrides

Any detected value can be replaced for a single invocation:

```bash
MAX_JOBS=32 bash research/build.sh                  # a large workstation
TORCH_CUDA_ARCH_LIST="9.0" bash research/build.sh   # cross-compile for H100
ARCH_PTX=1 bash research/build.sh                   # also embed PTX (see below)
BUILD_TEST=1 bash research/build.sh                 # also build C++ test binaries
USE_CUDA=0 bash research/build.sh                   # CPU-only build
ASSUME_YES=1 bash research/build.sh                 # no confirmation prompt
```

`ARCH_PTX=1` appends `+PTX` to the highest architecture, embedding intermediate
code that can be JIT-compiled onto a *newer* GPU than the build host. It costs
build time and binary size, so it is off by default; enable it when one build has
to serve machines with different GPUs.

Two flags are deliberately not left to chance. `BUILD_TEST=0` skips the C++ test
binaries, a large share of build time that is not needed to use or modify PyTorch
from Python. `USE_KINETO=1` stays on: Kineto is the CUPTI-based backend behind
`torch.profiler`, and kernel-level measurement is the point of this work.

## Version pins

Three pins are needed because PyTorch 2.3.0 predates breaking changes elsewhere:

- **`cmake<4`** — CMake 4.0 removed support for
  `cmake_minimum_required(VERSION < 3.5)`, which `third_party/protobuf` and other
  bundled submodules still declare. The configure step fails within a minute with
  `Compatibility with CMake < 3.5 has been removed from CMake`. PyTorch 2.3.0
  needs CMake >= 3.18, so any 3.x from 3.18 up works.

- **`numpy<2`** — 2.3.0 was compiled against the NumPy 1.x C ABI. NumPy 2.x gives
  `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`.
- **`setuptools==69.5.1`** — setuptools >= 80 removed `setup.py develop`, the
  build entry point. The alternative is `pip install --no-build-isolation -v -e .`

`build.sh check` reports all three.

## Rebuilding after editing the source

```bash
bash research/build.sh          # incremental
```

Measured on the reference machine: touching one CUDA source
(`aten/src/ATen/native/cuda/ActivationGeluKernel.cu`) recompiled it and relinked
the 213 MB `libtorch_cuda.so` in **about 20 seconds**, against ~2 hours for the
first build.

`setup.py develop` filters ninja's output, so the build log shows no
`Building`/`Linking` lines even when work was done. To confirm a rebuild really
happened, compare timestamps rather than reading the log:

```bash
ls -la --time-style=+%T aten/src/ATen/native/cuda/ActivationGeluKernel.cu
find build -name ActivationGeluKernel.cu.o -exec ls -la --time-style=+%T {} +
find build -name libtorch_cuda.so -exec ls -la --time-style=+%T {} +
```

The object and the library must be newer than the source. A rebuild that
finishes in seconds with the object *unchanged* means ninja saw no work — usually
because the edited path does not exist. `touch` on a wrong path silently creates
a new file instead of failing, so that mistake is easy to miss. PyTorch 2.3.0
splits activations into `Activation*Kernel.cu` files; there is no
`aten/src/ATen/native/cuda/Activation.cu`.

Editing a Python file under `torch/` needs no rebuild — `setup.py develop`
installs a link to this tree. Cleaning, from cheapest to most destructive:

```bash
python setup.py clean                        # drop build artifacts
rm -rf build/                                # force a full CMake reconfigure
git clean -xfd && git submodule foreach --recursive git clean -xfd
```

`ccache -s` shows the cache hit rate; a good rate is what makes rebuilds cheap.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error: pybind11/pybind11.h: No such file` | submodules not initialised | `git submodule update --init --recursive` |
| `unsupported GNU version! gcc versions later than N are not supported` | host GCC newer than this CUDA accepts | `sudo apt install -y gcc-N g++-N`; `build.sh` then selects it |
| build dies silently; `dmesg` shows `Out of memory: Killed process ... ld` | too many parallel jobs | lower `MAX_JOBS`, or add swap |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | NumPy 2 installed | `pip install "numpy<2"` |
| `Compatibility with CMake < 3.5 has been removed` | CMake 4.x | `conda install -y --override-channels -c conda-forge "cmake<4"`, then `rm -rf build/` |
| `error: invalid command 'develop'` | setuptools >= 80 | `pip install "setuptools==69.5.1"` |
| `ModuleNotFoundError: No module named 'torch._C'` | `import torch` run from inside the source tree | run from another directory; `build.sh verify` does this |
| `nvcc: command not found` | no toolkit on `PATH` | install CUDA, or set `CUDA_HOME` |
| `torch.cuda.is_available()` is `False` though the build succeeded | driver missing or mismatched, or `nouveau` loaded | `nvidia-smi`; install the proprietary driver |
| `CUDA error: no kernel image is available for execution on the device` | built for a different architecture | rebuild; `build.sh` detects the right one |
| `bad interpreter: /usr/bin/env bash^M` | script saved with CRLF line endings | `sed -i 's/\r$//' research/*.sh` |

`torch.__version__` reports **`2.3.0a0+git<sha>`**, not a plain `2.3.0`. That is
how a source build identifies itself; the suffix is not a failure.

### The build looks frozen about a third of the way through

It is not. Somewhere around 35 % of the ninja steps — step 2900 of 8290 on the
reference machine, though the total depends on `BUILD_TEST`, `USE_NCCL` and how
many architectures are being built — come the FlashAttention kernels, the slowest
files in the tree: 10-20 minutes each is normal. The question is whether the
machine is compiling slowly or thrashing. Run `vmstat 5` in a second terminal and
read the `si`/`so` (swap in/out) and `wa` (I/O wait) columns:

| Reading | Meaning | Action |
|---|---|---|
| `wa` < 10 %, `si`/`so` near 0 | compiling, just slow | wait |
| `si` high, `so` ≈ 0 | paging back in after an earlier spike | wait, it settles |
| `si` **and** `so` both tens of thousands, `wa` > 30 % | thrashing | `Ctrl+C`, rerun with a lower `MAX_JOBS` |
| swap more than ~80 % used | about to be OOM-killed | stop now |

Stopping costs little: ninja writes each `.o` as it finishes, so only the jobs
in flight are lost and the rerun resumes from there.

On the reference machine, `MAX_JOBS=8` gave `si 38633 / so 33988`, `wa 45 %` and
only 7 % user CPU — the build was moving data, not compiling. At `MAX_JOBS=3` the
same host showed `wa 0-5 %` with swap nearly untouched.

## Recording a machine

After a successful build, capture what it was verified on:

```bash
python torch/utils/collect_env.py > research/env-$(hostname).txt
bash research/build.sh check >> research/env-$(hostname).txt
conda list > research/conda-env-$(hostname).txt
```

`collect_env.py` is PyTorch's own reporting script. Naming both files after the
host lets several machines be recorded side by side instead of overwriting each
other. Add a row to the table below and commit all three changes together.

## Verified configurations

| Host | GPU | Arch | OS | CUDA | cuDNN | Python | `MAX_JOBS` | Build time |
|---|---|---|---|---|---|---|---|---|
| DESKTOP-2ASJTAS (i7-11800H, 16 GB) | RTX 3050 Ti Laptop, 4 GB | 8.6 | Ubuntu 22.04.5 on WSL2 | 12.1 | 8.9.7 | 3.10.21 | 5 | ≈2 h |

Verified 2026-09-20 with `build.sh verify`: `torch 2.3.0a0+git97ff6cf` (the
v2.3.0 tag commit), CUDA 12.1, cuDNN 8907, `sm_86` detected, CUDA matmul,
autograd and `torch.profiler` all working. Disk used: 2.0 GB in `build/`, 8.6 GB
for the whole tree.

The ≈2 h figure is the final run at `MAX_JOBS=5`, which reused about 2900 objects
compiled during an earlier aborted attempt. A clean build at `MAX_JOBS=5` on this
host would take longer — budget 3 hours for 16 GB of RAM with no warm `build/`.
