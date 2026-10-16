# Building PyTorch v2.3.0 from Source

This repository — `hungtruongOwolf/pytorch`, branch `research/base-v2.3.0` — is a
fork of PyTorch pinned to the v2.3.0 tag, so that its C++/CUDA internals can be
modified. The compiled library is machine-specific and is never committed, so
every machine has to build it once. Three files in `research/` do the work:

| File | Job |
|---|---|
| `setup.sh` | One-time install of everything the build needs: system packages, a conda environment, CUDA 12.1 and cuDNN 8.9.7 under `$HOME` (no root), the git submodules. Only the apt step needs `sudo`. Safe to re-run. |
| `build.sh` | Detects the GPU architecture, CUDA version, host compiler and job count from the host it runs on, then builds. **Never installs anything.** |
| `verify.py` | Run by `build.sh verify`: per-GPU matmul, autograd, profiler, and a real all-reduce across processes (gloo, and NCCL when there are two or more GPUs). |

Nothing here needs editing for a different GPU.

This repository holds only the PyTorch build. The distributed-training experiments that
run on top of it (Megatron-LM, pipeline parallelism on two nodes) live in a separate
repository, [pipeline-parallel-training](https://github.com/hungtruongOwolf/pipeline-parallel-training):
build PyTorch here first, then use that one.

## Start here

| Step | Command | How long |
|---|---|---|
| 1 | Check the *Requirements* table below | 2 min |
| 2 | `bash research/setup.sh check` — report what is present and what is missing, changes nothing | 1 min |
| 3 | `bash research/setup.sh` — install whatever is missing | mostly downloading (CUDA and cuDNN are several GB) |
| 4 | `bash research/build.sh` — shows the plan, asks, then builds | 45 min on a 48-core server (`MAX_JOBS=16`); about 2 h on a 16 GB laptop (`MAX_JOBS=5`) |
| 5 | `bash research/build.sh verify` | 1 min |

The only thing `setup.sh` does not install is the **NVIDIA driver**: `nvidia-smi`
must already work. Everything else — conda, CUDA, cuDNN, the Python environment —
it installs itself, in `$HOME`.

Everything from *Version pins* onward is reference material — read it when
something goes wrong or when you want to change a default, not on the way through.

## Requirements

| | |
|---|---|
| OS | x86_64 Linux — native, on a cluster, or under WSL2. Verified on Ubuntu 22.04 (WSL2) and Ubuntu 24.04 (bare metal). `setup.sh`'s apt step assumes Debian/Ubuntu; on another distro it prints the packages to install by hand |
| GPU | NVIDIA, with the proprietary driver loaded — `nvidia-smi` must work |
| Driver | >= 530.30.02 for CUDA 12.1, or >= 520.61.05 for CUDA 11.8 |
| CUDA Toolkit | 11.8 or 12.1 (the versions PyTorch 2.3.0 officially targets). `setup.sh` installs 12.1.1 under `$HOME/cuda-12.1`, without root and without touching the system CUDA or the driver. `build.sh` refuses any other version unless `ALLOW_UNTESTED_CUDA=1` |
| cuDNN | 8.9.x matching the CUDA version. `setup.sh` unpacks 8.9.7 into that toolkit |
| Host compiler | GCC 12 or older for CUDA 12.1. Ubuntu 24.04 defaults to GCC 13, which nvcc rejects: `setup.sh` installs `gcc-12`/`g++-12` when the default is newer, and `build.sh` selects it |
| Python | 3.10, in the conda environment `pt230` that `setup.sh` creates |
| RAM | 32 GB comfortable; 16 GB workable but limits `MAX_JOBS` to about 5; 8 GB only with generous swap |
| Disk | 50 GB minimum, 100 GB comfortable, on **ext4** — not NTFS |
| Sudo | only for the apt step (passwordless `sudo` or root); without it the step lists the packages to install by hand |

`setup.sh check` and `build.sh check` inspect all of this and report what is
missing, changing nothing. `build.sh` itself runs the same inspection and refuses
to start the build while anything is marked `[FAIL]`.

## Quick start

### A. Using this repository

This is the path for anyone who just wants to build what is already here. The
branch `research/base-v2.3.0` already contains both the v2.3.0 source tree and
this `research/` directory.

```bash
# 1. Source
git clone https://github.com/hungtruongOwolf/pytorch.git && cd pytorch
git checkout research/base-v2.3.0            # the branch already exists remotely

# 2. One-time machine setup (re-running is safe: it redoes only what is missing)
bash research/setup.sh check                 # read-only report
bash research/setup.sh                       # install what is missing, then check again

# 3. Build
tmux new -s ptbuild                          # survive a dropped terminal
bash research/build.sh check                 # inspect the host, change nothing
bash research/build.sh                       # shows the plan, asks, then builds
bash research/build.sh verify
```

Do **not** check out the `v2.3.0` tag directly. The tag predates this directory,
so the working tree would have no `research/build.sh` to run.

`setup.sh` writes `~/.pt230-env.sh` (which CUDA, which conda environment) and
`build.sh` reads it, so a fresh shell or a new tmux window needs no manual
`conda activate`. To use the build interactively afterwards, run
`source ~/.pt230-env.sh`.

`setup.sh` can also run one step at a time — `apt`, `conda`, `env`, `cuda`,
`cudnn`, `source`, in that order — for example `bash research/setup.sh cuda`.
Each step looks at the real state of the machine before and after (never at a
"done" marker file) and verifies its downloads by checksum.

`build.sh` prints its plan and waits for confirmation first, so a misdetected
setting costs seconds rather than hours. It also compiles a tiny test kernel with
the real nvcc, host compiler and target architecture before the long build.

Doing the setup by hand instead — useful on a cluster with `module load`, or to
see what `setup.sh` does — is described under *Manual prerequisites* in the
Reference.

### B. Recreating the branch from upstream

Only needed to start a fresh branch from the v2.3.0 tag — for example on a new
fork. This is how `research/base-v2.3.0` was created:

```bash
git clone https://github.com/<your-fork>/pytorch.git && cd pytorch
git remote add upstream https://github.com/pytorch/pytorch.git
git fetch upstream --tags                    # forks copy the default branch only,
                                             # so the v2.3.0 tag is not inherited
git checkout -b research/base-v2.3.0 v2.3.0
git submodule sync --recursive
git submodule update --init --recursive
```

Then copy this `research/` directory in and commit it. Pushing a branch based on
an old tag needs a token with the `workflow` scope, because it creates files
under `.github/workflows/`. (A fine-grained token with *Contents: read and write*
is enough for pushing later commits to a branch that already exists.)

## Version pins

Three pins are needed because PyTorch 2.3.0 predates breaking changes elsewhere.
`setup.sh` applies all of them when it creates the environment:

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
| `unsupported GNU version` on Ubuntu 24.04 | the default GCC is 13; CUDA 12.1 accepts up to 12 | `bash research/setup.sh` installs `gcc-12`; `build.sh` selects it. Export `CC`/`CXX`/`CUDAHOSTCXX` as `gcc-12`/`g++-12` too before anything that JIT-compiles C++ later (for example Megatron-LM's dataset helper) |
| `build.sh` stops with `CUDA x.y found ..., but PyTorch 2.3.0 targets 11.8 and 12.1` | another CUDA is first on `PATH` | run `bash research/setup.sh` (it records the right one in `~/.pt230-env.sh`), or `ALLOW_UNTESTED_CUDA=1` at your own risk |
| `cuDNN headers not found` | cuDNN missing or not next to the toolkit | `bash research/setup.sh cudnn`, or `USE_CUDNN=0` to build without it |

Searching a build log for `error:` gives false positives (for example
`system_error:41` in a normal line); search for `FAILED:` instead.

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

---

# Reference

The sections above are the path through a first build, in order. What follows is
looked up as needed.

## What setup.sh does

Run it once per machine. Each step first looks at the real state of the machine, so
re-running is always safe and redoes only what is missing.

| Step | Installs | Needs root |
|---|---|---|
| `apt` | compilers, git, tmux, ccache; `gcc-12` only if the default GCC is newer than 12 | yes (passwordless `sudo`) |
| `conda` | Miniforge (conda-forge only, so no Anaconda terms-of-service prompt) in `$HOME/miniforge3` | no |
| `env` | conda environment `pt230`: Python 3.10, `numpy<2`, `setuptools==69.5.1`, `cmake<4`, ninja, `mkl-static` | no |
| `cuda` | CUDA Toolkit 12.1.1 from the NVIDIA `.run` installer into `$HOME/cuda-12.1` — toolkit only, no driver | no |
| `cudnn` | cuDNN 8.9.7 for CUDA 12, unpacked into that toolkit | no |
| `source` | the git submodules of this checkout | no |

`bash research/setup.sh check` reports each step as present or missing and changes
nothing. Settings (environment variables): `CONDA_ENV=pt230`,
`CONDA_ROOT=$HOME/miniforge3`, `CUDA_PREFIX=$HOME/cuda-12.1`,
`SETUP_DIR=$HOME/.cache/pt230-setup`, `ENV_FILE=$HOME/.pt230-env.sh`,
`KEEP_DOWNLOADS=1` (keep the CUDA/cuDNN archives), `INSTALL_MAGMA=1` (also install
`magma-cuda121`; optional, only some `torch.linalg` CUDA operations need it).

## What build.sh detects

| Setting | Source |
|---|---|
| `TORCH_CUDA_ARCH_LIST` | `nvidia-smi --query-gpu=compute_cap`, deduplicated and sorted |
| `MAX_JOBS` | `min(nproc, RAM_GB / 2, JOBS_CAP)` with `JOBS_CAP=16` by default — PyTorch needs ~2 GB of RAM per compile job, link steps peak higher, and on a 48-core host more than 16 jobs gains little |
| `CUDA_HOME` | explicit value, else the one `setup.sh` recorded in `~/.pt230-env.sh`, else `nvcc` on `PATH`, else the newest `/usr/local/cuda-*` |
| `CC`, `CXX`, `CUDAHOSTCXX` | the newest installed `gcc-N` that the detected CUDA accepts (`gcc-12` for CUDA 12.1) |
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

### Building where no GPU is visible

A cluster login node usually has no GPU, so architecture detection cannot work
and `build.sh` stops. Compiling there is still legitimate — name the target
architecture explicitly and it proceeds with a warning:

```bash
TORCH_CUDA_ARCH_LIST="8.0" bash research/build.sh     # A100
TORCH_CUDA_ARCH_LIST="9.0" bash research/build.sh     # H100
```

Look the value up with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
on a compute node, or from NVIDIA's compute-capability table. Getting it wrong
produces a build that finishes and then fails at run time with `no kernel image
is available for execution on the device`.

Run `build.sh verify` on a node that has the GPU, not on the login node.

`MAX_JOBS` is a ceiling, not a target to beat. The 2 GB-per-job estimate holds
for ordinary sources, but the FlashAttention kernels under
`aten/src/ATen/native/transformers/cuda/flash_attn/` take **3-6 GB each** — they
are CUTLASS templates, and nvcc has to expand the whole instantiation tree in
memory. On a 16 GB host, raising `MAX_JOBS` to 8 puts the machine into swap
thrashing during that stretch and makes the build *slower*, not faster. See
*The build looks frozen about a third of the way through*, under Troubleshooting,
for how to tell thrashing from slow compiling.

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

## Manual prerequisites

`setup.sh` does all of this automatically, except the NVIDIA driver. This section
is for doing it by hand, for a cluster where software comes from `module load`, and
for the WSL2-only settings at the end, which no script can change.

Five things must exist before the build. Each is shown for the three
environments this is expected to run in:

| | |
|---|---|
| **Native Linux** | a workstation or server where you have `sudo` |
| **Cluster** | a shared Linux machine with no root, software via `module load` |
| **WSL2** | Ubuntu inside Windows |

Check each item before installing anything — on a cluster, most are already
provided.

### 1. NVIDIA driver

Check: `nvidia-smi` must print a table of GPUs.

**Native Linux** — if it prints nothing, or `lsmod | grep nouveau` shows the
open-source driver is loaded, CUDA cannot work:

```bash
ubuntu-drivers devices              # shows the recommended driver
sudo ubuntu-drivers install         # or: sudo apt install nvidia-driver-550
sudo reboot
```

If the module still will not load afterwards, Secure Boot is blocking it: enroll
the MOK key when prompted at the next boot, or disable Secure Boot in the UEFI
settings.

**WSL2** — install the driver **on Windows**, never inside WSL. The GPU is
borrowed from the Windows driver, which is why `nvidia-smi` works in WSL with no
Linux driver present. Installing a Linux driver inside WSL breaks the passthrough
and the distribution has to be reinstalled.

**Cluster** — already installed. If `nvidia-smi` fails on a login node that is
expected, since login nodes usually have no GPU; see *Building where no GPU is
visible* below.

### 2. Build tools

`git`, a C++ compiler, `wget`, `tmux` and `ccache`.

```bash
# Native Linux and WSL2:
sudo apt install -y build-essential git wget tmux ccache

# Cluster, typically:
module load gcc git
```

`ccache` is optional but makes every rebuild after the first far cheaper. Give it
room once installed: `ccache -M 25G`.

### 3. CUDA Toolkit 11.8 or 12.1

Check: `nvcc --version`. These are the versions PyTorch 2.3.0 officially targets.

**Native Linux** (Ubuntu 22.04 shown; substitute your distro's repository path):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1        # NOT the "cuda" metapackage
```

**WSL2** — identical, except the repository path is `wsl-ubuntu`, which ships no
driver packages:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1
```

In both cases install `cuda-toolkit-12-1`, never the `cuda` metapackage: that one
pulls in a driver and may downgrade the one already installed. Under WSL2 that
would break the GPU passthrough outright.

**Cluster** — `module load cuda/12.1`. `build.sh` finds it on `PATH`, no further
configuration needed.

After a manual install, put it on `PATH` permanently:

```bash
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### 4. cuDNN 8.9.x

Matching the CUDA major version. PyTorch 2.3.0 supports the 8.x line only —
cuDNN 9 needs PyTorch 2.4 or newer.

Check: `grep CUDNN_MAJOR /usr/local/cuda-12.1/include/cudnn_version.h`

**Native Linux**, with the CUDA apt repository from step 3 configured:

```bash
sudo apt install -y libcudnn8 libcudnn8-dev
```

**WSL2** — `libcudnn8` is not in the `wsl-ubuntu` repository. Use NVIDIA's public
redistributable, which needs no account, and unpack it next to CUDA so cmake
finds it there:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp -P cudnn-*/include/cudnn*.h /usr/local/cuda-12.1/include/
sudo cp -P cudnn-*/lib/libcudnn*    /usr/local/cuda-12.1/lib64/
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn*.h /usr/local/cuda-12.1/lib64/libcudnn*
sudo ldconfig
```

`cp -P` is required: it preserves the `libcudnn.so → .so.8 → .so.8.9.7` symlink
chain, which a plain copy would flatten and break. Use the `_cuda11` tarball
instead if building against CUDA 11.8.

**Cluster** — usually provided with the CUDA module. Without root and without a
module, unpack the same redistributable under `$HOME` rather than
`/usr/local/cuda-12.1`, and point cmake at it with `CUDNN_ROOT`,
`CUDNN_INCLUDE_DIR` and `CUDNN_LIBRARY_PATH`. That path has not been tested here;
`USE_CUDA=1 USE_CUDNN=0 bash research/build.sh` is the fallback if it gives
trouble — the build works without cuDNN, with slower convolutions.

### 5. conda

Check: `conda --version`. Same on all three environments:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
```

`-b` runs the installer unattended instead of paging through the licence. On a
cluster, check for a conda or Miniforge module first.

### Extra steps on WSL2 only

Three further things apply under WSL2 and nowhere else. `build.sh` detects WSL
and warns about them, but the fixes are manual.

**Give WSL more RAM.** It takes half the host's by default, which starves
`MAX_JOBS`. Create `C:\Users\<name>\.wslconfig` on the Windows side, then run
`wsl --shutdown` in PowerShell and reopen Ubuntu:

```ini
[wsl2]
memory=12GB
swap=24GB
processors=16
```

The large swap matters more than the memory: it is what keeps a peak link step
from being OOM-killed an hour into a build. Confirm with `free -h` afterwards.

**Keep the repository in the Linux filesystem** (`~/pytorch`). Building from
`/mnt/c` or `/mnt/e` is 10-50x slower through the translation layer, and NTFS is
case-insensitive, which breaks the build outright.

**Kernel profiling is restricted.** CUPTI and Nsight hardware counters are limited
under WSL2. Building is unaffected and `torch.profiler` still returns entries,
but measuring a modified kernel properly eventually needs native Linux.

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
| Chameleon `pytorch-dev1` (48 cores, 125 GB) | 2 x Tesla P100-PCIE-16GB | 6.0 | Ubuntu 24.04, bare metal | 12.1.1 (user-space) | 8.9.7 | 3.10.22 | 16 | 45 min |
| Chameleon `pytorch-dev2` (48 cores, 125 GB) | 2 x Tesla P100-PCIE-16GB | 6.0 | Ubuntu 24.04, bare metal | 12.1.1 (user-space) | 8.9.7 | 3.10.22 | 16 | 45 min |

Laptop, verified 2026-09-20 with `build.sh verify`: `torch 2.3.0a0+git97ff6cf` (the
v2.3.0 tag commit), CUDA 12.1, cuDNN 8907, `sm_86` detected, CUDA matmul,
autograd and `torch.profiler` all working. Disk used: 2.0 GB in `build/`, 8.6 GB
for the whole tree.

The ≈2 h figure is the final run at `MAX_JOBS=5`, which reused about 2900 objects
compiled during an earlier aborted attempt. A clean build at `MAX_JOBS=5` on this
host would take longer — budget 3 hours for 16 GB of RAM with no warm `build/`.

Chameleon nodes, verified 2026-10-15 with `setup.sh` followed by
`build.sh` and `build.sh verify`, with no manual fix in between: `torch
2.3.0a0+git7db8f73` (commit `7db8f73d39` of this branch), CUDA 12.1, cuDNN 8907,
NCCL 2.20.5, GCC 12.4, `sm_60` detected on both GPUs. fp32 and fp16 matmul,
autograd, the profiler, and gloo and NCCL all-reduce across the two GPUs all
passed. The same scripts were also tested from scratch on a clean Ubuntu 24.04
WSL2 instance. The distributed training run that uses this build is in
[pipeline-parallel-training](https://github.com/hungtruongOwolf/pipeline-parallel-training).
