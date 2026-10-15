#!/usr/bin/env bash
#
# Build PyTorch v2.3.0 from source on any Linux host with an NVIDIA GPU.
#
#   bash research/build.sh check      inspect this host, change nothing
#   bash research/build.sh            inspect, show the plan, confirm, then build
#   bash research/build.sh verify     check that an existing build works
#
# This script never installs anything. On a new machine run
# 'bash research/setup.sh' once first: it installs CUDA 12.1, cuDNN, conda and the
# rest, and writes ~/.pt230-env.sh, which this script reads.
#
# Every machine-dependent setting is detected from this host, so no edit is
# needed when moving to a different GPU. Any setting can be overridden:
#
#   MAX_JOBS=32 bash research/build.sh        (default is min(CPUs, RAM/2, JOBS_CAP=16))
#   ALLOW_UNTESTED_CUDA=1 bash research/build.sh   build with a CUDA other than 11.8/12.1
#   TORCH_CUDA_ARCH_LIST="9.0" bash research/build.sh
#   ARCH_PTX=1 bash research/build.sh        also embed PTX for newer GPUs
#   BUILD_TEST=1 bash research/build.sh      also build the C++ test binaries
#   ASSUME_YES=1 bash research/build.sh      skip the confirmation prompt

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-build}"

# Disk budget: source + submodules (~3 GB), build artifacts for one architecture
# (~10-20 GB), conda env (~5 GB), CUDA toolkit (~6 GB), ccache (~10-25 GB).
DISK_MIN_GB=50
DISK_GOOD_GB=100
RAM_MIN_GB=8

PASS=0; WARN=0; FAIL=0
ok()    { printf '  [ ok ]  %s\n' "$*"; PASS=$((PASS+1)); }
warn()  { printf '  [warn]  %s\n' "$*"; WARN=$((WARN+1)); }
bad()   { printf '  [FAIL]  %s\n' "$*"; FAIL=$((FAIL+1)); }
group() { printf '\n%s\n' "$*"; }
die()   { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

case "$MODE" in
    check|build|verify) ;;
    -h|--help|help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown mode '$MODE' (expected: check, build, verify)" ;;
esac

# Read the choices 'setup.sh' recorded (which CUDA, which conda environment), so a
# bare shell or a fresh tmux window needs no manual 'conda activate'. Without the
# file, everything below falls back to detecting from PATH, as before.
ENV_FILE="${ENV_FILE:-$HOME/.pt230-env.sh}"
load_env_file() {
    [ -r "$ENV_FILE" ] || return 0
    # conda's activate scripts reference unset variables, which 'set -u' rejects.
    set +u
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set -u
}

# =============================================================== detection ===

detect_os() {
    group "Operating system"
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        printf '  %s, kernel %s, %s\n' "${PRETTY_NAME:-unknown}" "$(uname -r)" "$(uname -m)"
        case "${ID:-}" in
            ubuntu|debian) ok "Debian-family distribution" ;;
            *) warn "Not Ubuntu/Debian: translate the apt commands in BUILD.md to this distro" ;;
        esac
    else
        warn "Cannot read /etc/os-release"
    fi
    if grep -qi microsoft /proc/version 2>/dev/null; then
        warn "Running under WSL: CUPTI/Nsight kernel profiling is limited, and the"
        warn "repository must live under the Linux home directory, not /mnt/*"
    fi
}

detect_cpu_ram() {
    group "CPU and memory"
    CPUS="$(nproc 2>/dev/null || echo 0)"
    RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)"
    SWAP_GB="$(awk '/SwapTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)"
    printf '  %s CPU threads, %s GB RAM, %s GB swap\n' "$CPUS" "$RAM_GB" "$SWAP_GB"

    # PyTorch needs roughly 2 GB of RAM per parallel compile job, and link steps
    # peak higher. RAM, not core count, is the binding constraint: oversubscribe
    # it and the linker gets OOM-killed an hour into the build.
    if [ -z "${MAX_JOBS:-}" ]; then
        local by_ram=$(( RAM_GB / 2 )) cap="${JOBS_CAP:-16}"
        MAX_JOBS=$(( by_ram < CPUS ? by_ram : CPUS ))
        # A 48-core server with 125 GB would otherwise get 48 jobs. The heaviest
        # sources (FlashAttention, 3-6 GB each on the laptop) could then exhaust RAM
        # on a host with no swap, and the OOM killer ends the build without a message.
        [ "$MAX_JOBS" -gt "$cap" ] && MAX_JOBS="$cap"
        [ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
    fi
    export MAX_JOBS

    if [ "$RAM_GB" -lt "$RAM_MIN_GB" ]; then
        bad "Under ${RAM_MIN_GB} GB RAM: expect the linker to be OOM-killed"
    elif [ "$RAM_GB" -lt 16 ] && [ "$SWAP_GB" -lt 8 ]; then
        warn "RAM is tight and there is little swap; see BUILD.md troubleshooting"
    else
        ok "Memory is sufficient for MAX_JOBS=$MAX_JOBS"
    fi
}

detect_disk() {
    group "Disk"
    local avail fstype
    avail="$(df -BG --output=avail "$REPO_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    fstype="$(df --output=fstype "$REPO_ROOT" 2>/dev/null | tail -1 | tr -d ' ')"
    printf '  %s: %s GB free (%s)\n' "$REPO_ROOT" "${avail:-?}" "${fstype:-?}"
    case "$fstype" in
        fuseblk|ntfs|ntfs3|vfat|exfat)
            bad "Repository is on $fstype, which is case-insensitive and slow. Move it to ext4." ;;
    esac
    if [ -z "$avail" ]; then
        warn "Cannot determine free space"
    elif [ "$avail" -ge "$DISK_GOOD_GB" ]; then
        ok "Ample space for source, build artifacts and ccache"
    elif [ "$avail" -ge "$DISK_MIN_GB" ]; then
        warn "Workable but tight; keep the ccache small (ccache -M 10G)"
    else
        bad "Under ${DISK_MIN_GB} GB free: free space before building"
    fi
}

detect_gpu() {
    group "NVIDIA driver and GPU"
    if lsmod 2>/dev/null | grep -q '^nouveau'; then
        bad "The open-source 'nouveau' driver is loaded; CUDA cannot work"
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        GPU_DESC="none visible"
        # No GPU is only fatal when the architecture has to be detected. A cluster
        # login node has no GPU but can still compile, provided the caller says
        # which architecture to target.
        if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
            warn "No GPU visible here, but TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST was given, so compiling anyway"
            warn "The result cannot be tested on this host: run 'build.sh verify' where a GPU is present"
            NGPU=0
            [ -z "${USE_NCCL:-}" ] && export USE_NCCL=0
        else
            bad "nvidia-smi is not working: no usable NVIDIA driver, and no TORCH_CUDA_ARCH_LIST to compile against"
            bad "On a GPU machine, install the driver. On a login node, pass the target architecture, e.g. TORCH_CUDA_ARCH_LIST=8.0"
        fi
        return
    fi

    local driver
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    GPU_DESC="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do printf '  GPU %s\n' "$line"; done
    ok "Driver $driver"

    # CUDA 12.1 requires driver >= 530.30.02; CUDA 11.8 requires >= 520.61.05.
    local drv_major="${driver%%.*}"
    if [ -n "$drv_major" ] && [ "$drv_major" -lt 530 ] 2>/dev/null; then
        warn "Driver $driver predates CUDA 12.1's minimum (530.30.02); upgrade it or use CUDA 11.8"
    fi

    # Compile device code only for the architectures actually present. Left
    # unset, nvcc builds every architecture PyTorch supports: several times
    # slower, and none of the extra output is usable here.
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        local caps
        caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -u -V | paste -sd' ' -)"
        if [ -n "$caps" ]; then
            if [ "${ARCH_PTX:-0}" = "1" ]; then
                # Append +PTX to the highest architecture (the list is sorted
                # ascending) so the binary can also JIT onto a newer GPU.
                TORCH_CUDA_ARCH_LIST="$(printf '%s' "$caps" | sed 's/\([0-9.]\+\)$/\1+PTX/')"
            else
                TORCH_CUDA_ARCH_LIST="$caps"
            fi
            export TORCH_CUDA_ARCH_LIST
            ok "Will compile for architecture(s): $TORCH_CUDA_ARCH_LIST"
        else
            warn "Driver too old to report compute_cap; set TORCH_CUDA_ARCH_LIST manually"
        fi
    else
        ok "Using the supplied TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    fi

    # NCCL only matters with more than one GPU, and building it costs time.
    if [ -z "${USE_NCCL:-}" ]; then
        [ "${NGPU:-0}" -gt 1 ] && export USE_NCCL=1 || export USE_NCCL=0
    fi
}

detect_cuda() {
    group "CUDA toolkit"
    if [ -z "${CUDA_HOME:-}" ]; then
        if command -v nvcc >/dev/null 2>&1; then
            CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        elif [ -x /usr/local/cuda/bin/nvcc ]; then
            CUDA_HOME=/usr/local/cuda
        else
            CUDA_HOME="$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d 2>/dev/null | sort -V | tail -1)"
        fi
    fi

    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        CUDA_VERSION="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
        export USE_CUDA=1
        ok "nvcc $CUDA_VERSION at $CUDA_HOME"
        case "$CUDA_VERSION" in
            11.8|12.1) ok "CUDA $CUDA_VERSION is officially supported by PyTorch 2.3.0" ;;
            *)
                if [ "${ALLOW_UNTESTED_CUDA:-0}" = "1" ]; then
                    warn "CUDA $CUDA_VERSION is not one PyTorch 2.3.0 targets (11.8, 12.1); continuing because ALLOW_UNTESTED_CUDA=1"
                else
                    bad "CUDA $CUDA_VERSION found at $CUDA_HOME, but PyTorch 2.3.0 targets 11.8 and 12.1. Run 'bash research/setup.sh' (installs 12.1 beside it), or set ALLOW_UNTESTED_CUDA=1"
                fi ;;
        esac
    else
        CUDA_VERSION=""
        export USE_CUDA=0
        bad "No CUDA toolkit found; install one or set CUDA_HOME (BUILD.md requirements)"
    fi

    # -L is required: a CUDA apt install makes <cuda>/include a symlink to
    # targets/<arch>/include, and find does not follow symlinks given as
    # arguments unless told to.
    local cudnn_h
    # CUDA_HOME comes first: setup.sh unpacks cuDNN into the toolkit it installed.
    cudnn_h="$(find -L "${CUDA_HOME:-/nonexistent}/include" /usr/include /usr/local/cuda*/include -maxdepth 1 -name cudnn_version.h 2>/dev/null | head -1)"
    if [ -n "$cudnn_h" ]; then
        ok "cuDNN $(awk '/define CUDNN_MAJOR|define CUDNN_MINOR|define CUDNN_PATCHLEVEL/ {printf "%s.", $3}' "$cudnn_h" | sed 's/\.$//')"
    elif [ "${USE_CUDNN:-1}" = "0" ]; then
        warn "cuDNN headers not found; building without cuDNN because USE_CUDNN=0"
    else
        bad "cuDNN headers not found: run 'bash research/setup.sh', or set USE_CUDNN=0 to build without it"
    fi
    export USE_CUDNN="${USE_CUDNN:-$USE_CUDA}"
}

detect_compiler() {
    group "Host compiler"
    command -v gcc >/dev/null 2>&1 || { bad "gcc not installed"; return; }
    local have full
    full="$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion)"
    have="${full%%.*}"
    printf '  default gcc %s\n' "$full"
    [ -z "${CUDA_VERSION:-}" ] && return

    # nvcc refuses a host GCC newer than a per-release maximum. The authoritative
    # table is in NVIDIA's CUDA Installation Guide for Linux, "Host Compiler
    # Support Policy"; update this mapping when moving to a newer CUDA.
    #   11.x -> 11 | 12.0-12.2 -> 12 | 12.3-12.5 -> 13 | 12.6+ -> 14
    local cu_major="${CUDA_VERSION%%.*}" cu_minor="${CUDA_VERSION##*.}" gcc_max
    if   [ "$cu_major" -le 11 ];                          then gcc_max=11
    elif [ "$cu_major" -eq 12 ] && [ "$cu_minor" -le 2 ]; then gcc_max=12
    elif [ "$cu_major" -eq 12 ] && [ "$cu_minor" -le 5 ]; then gcc_max=13
    else                                                       gcc_max=14
    fi

    if [ "$have" -le "$gcc_max" ]; then
        ok "gcc $have is accepted by CUDA $CUDA_VERSION (maximum $gcc_max)"
        return
    fi
    if [ -n "${CUDAHOSTCXX:-}" ]; then
        ok "Using the supplied CUDAHOSTCXX=$CUDAHOSTCXX"
        return
    fi
    local v
    for v in $(seq "$gcc_max" -1 8); do
        if command -v "gcc-$v" >/dev/null 2>&1 && command -v "g++-$v" >/dev/null 2>&1; then
            CC="$(command -v "gcc-$v")"; CXX="$(command -v "g++-$v")"; CUDAHOSTCXX="$CXX"
            export CC CXX CUDAHOSTCXX
            ok "gcc $have is too new for CUDA $CUDA_VERSION; using gcc-$v instead"
            return
        fi
    done
    bad "gcc $have exceeds CUDA $CUDA_VERSION's maximum ($gcc_max). Run: sudo apt install -y gcc-$gcc_max g++-$gcc_max"
}

# Compile and link one trivial kernel with the exact nvcc, host compiler and
# architecture the build will use. A toolchain mismatch (a host compiler or C
# library too new for this CUDA) shows up here in seconds, instead of an hour into
# the build.
detect_nvcc_smoke() {
    group "CUDA compile test"
    [ -n "${CUDA_VERSION:-}" ] || { warn "No CUDA toolkit, skipping"; return; }
    local arch tmp host
    arch="${TORCH_CUDA_ARCH_LIST:-}"; arch="${arch%% *}"; arch="${arch%%+*}"; arch="${arch//./}"
    if [ -z "$arch" ]; then warn "No target architecture known, skipping"; return; fi
    tmp="$(mktemp -d)"
    cat >"$tmp/t.cu" <<'CU'
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
// rsqrtf and cospif are the functions CUDA's math headers and a newer C library
// have been known to disagree about; using them forces that header path.
__global__ void k(float* x) { x[0] = rsqrtf(x[0]) + cospif(x[0]); }
int main() {
    float* d = nullptr;
    if (cudaMalloc(&d, sizeof(float)) != cudaSuccess) { printf("no device\n"); return 2; }
    cudaMemset(d, 0, sizeof(float));
    k<<<1, 1>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 1;
}
CU
    host="${CUDAHOSTCXX:-$(command -v g++)}"
    if "$CUDA_HOME/bin/nvcc" -ccbin "$host" -gencode "arch=compute_${arch},code=sm_${arch}" \
            -o "$tmp/t" "$tmp/t.cu" >"$tmp/log" 2>&1; then
        ok "nvcc with $host compiles and links a kernel for sm_${arch}"
        if [ "${NGPU:-0}" -gt 0 ]; then
            if "$tmp/t" >"$tmp/run" 2>&1; then ok "...and the kernel runs on the GPU"
            else warn "The test kernel did not run ($(head -1 "$tmp/run")); expected only if the target architecture differs from the installed GPU"; fi
        fi
    else
        bad "nvcc with $host cannot compile a trivial kernel for sm_${arch}:"
        head -8 "$tmp/log" | sed 's/^/          /'
    fi
    rm -rf "$tmp"
}

detect_tooling() {
    group "Build tooling"
    local t
    for t in git ninja; do
        command -v "$t" >/dev/null 2>&1 && ok "$t present" || bad "$t missing"
    done

    # CMake 4.0 removed support for cmake_minimum_required(VERSION < 3.5), which
    # PyTorch 2.3.0's bundled protobuf and several other submodules still declare.
    # The configure step dies within a minute. PyTorch 2.3.0 needs >= 3.18.
    if command -v cmake >/dev/null 2>&1; then
        local cmv cmv_major cmv_minor
        cmv="$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
        cmv_major="${cmv%%.*}"
        cmv_minor="$(printf '%s' "$cmv" | cut -d. -f2)"
        if [ "${cmv_major:-0}" -ge 4 ] 2>/dev/null; then
            bad "cmake $cmv dropped pre-3.5 compatibility that third_party/protobuf still needs; install a 3.x: conda install -y --override-channels -c conda-forge \"cmake<4\""
        elif [ "${cmv_major:-0}" -eq 3 ] && [ "${cmv_minor:-0}" -lt 18 ] 2>/dev/null; then
            bad "cmake $cmv is older than the 3.18 minimum for PyTorch 2.3.0"
        else
            ok "cmake $cmv"
        fi
    else
        bad "cmake missing"
    fi
    if command -v ccache >/dev/null 2>&1; then
        ok "ccache present, max size $(ccache -p 2>/dev/null | awk -F'= ' '/max_size/ {print $2; exit}')"
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    else
        warn "ccache missing: incremental rebuilds will be far slower (sudo apt install -y ccache)"
    fi
    command -v tmux >/dev/null 2>&1 \
        || warn "tmux missing: a dropped terminal would kill a multi-hour build"
}

detect_python() {
    group "Python environment"
    if [ -z "${CONDA_PREFIX:-}" ]; then
        bad "No conda environment active: run 'bash research/setup.sh' once, or 'conda activate pt230'"
        return
    fi
    ok "conda env $CONDA_PREFIX"
    export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

    PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    case "$PY_VER" in
        3.8|3.9|3.10|3.11) ok "Python $PY_VER" ;;
        "") bad "python is not runnable in this environment" ;;
        *)  warn "Python $PY_VER is outside PyTorch 2.3.0's supported 3.8-3.11 range" ;;
    esac

    local np st
    np="$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"
    case "${np:-}" in
        "")  bad "numpy not installed: pip install -r requirements.txt" ;;
        1.*) ok "numpy $np" ;;
        *)   bad "numpy $np breaks PyTorch 2.3.0's C ABI: pip install \"numpy<2\"" ;;
    esac

    st="$(python -c 'import setuptools; print(setuptools.__version__)' 2>/dev/null)"
    if [ -n "${st:-}" ] && [ "${st%%.*}" -ge 80 ] 2>/dev/null; then
        bad "setuptools $st removed 'setup.py develop': pip install \"setuptools==69.5.1\""
    elif [ -n "${st:-}" ]; then
        ok "setuptools $st"
    fi
}

detect_source() {
    group "Source tree"
    if [ ! -f "$REPO_ROOT/setup.py" ] || [ ! -f "$REPO_ROOT/version.txt" ]; then
        bad "$REPO_ROOT does not look like a PyTorch checkout"
        return
    fi
    local ver
    ver="$(tr -d '[:space:]' < "$REPO_ROOT/version.txt")"
    # The release branch carries "2.3.0a0"; a plain "2.3.0" also counts.
    case "$ver" in
        2.3.0|2.3.0a0) ok "PyTorch $ver at $REPO_ROOT" ;;
        *)             warn "version.txt says $ver, expected 2.3.0a0" ;;
    esac
    # A leading '-' means a submodule was never fetched, which surfaces much
    # later as a missing-header error such as pybind11/pybind11.h.
    local missing
    missing="$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c '^-')"
    if [ "${missing:-0}" -gt 0 ]; then
        bad "$missing submodule(s) not initialised: git submodule update --init --recursive"
    else
        ok "Submodules initialised"
    fi
}

set_build_flags() {
    # Skip the C++ test binaries: a large share of build time, and not needed to
    # use or modify PyTorch from Python.
    export BUILD_TEST="${BUILD_TEST:-0}"
    # Kineto is the CUPTI-based backend behind torch.profiler, which is how
    # kernel-level changes get measured. It stays on.
    export USE_KINETO="${USE_KINETO:-1}"
    export USE_MKLDNN="${USE_MKLDNN:-1}"
    export USE_DISTRIBUTED="${USE_DISTRIBUTED:-1}"
}

# ================================================================== report ===

print_plan() {
    cat <<PLAN

------------------------------------------------------------------
Build plan
  GPU                   ${GPU_DESC:-unknown}  (arch ${TORCH_CUDA_ARCH_LIST:-ALL -- slow})
  CUDA                  ${CUDA_VERSION:-none}  at ${CUDA_HOME:-none}
  Host compiler         ${CXX:-$(command -v g++ 2>/dev/null || echo none)}
  Parallel jobs         ${MAX_JOBS}  (from ${CPUS:-?} threads, ${RAM_GB:-?} GB RAM)
  conda env             ${CONDA_PREFIX:-none}  (Python ${PY_VER:-?})
  USE_CUDA / USE_CUDNN  ${USE_CUDA:-0} / ${USE_CUDNN:-0}
  USE_NCCL / BUILD_TEST ${USE_NCCL:-0} / ${BUILD_TEST:-0}
  Source                ${REPO_ROOT}
  Checks                ${PASS} ok, ${WARN} warning(s), ${FAIL} failure(s)
------------------------------------------------------------------
PLAN
}

# =================================================================== build ===

run_build() {
    local log="$REPO_ROOT/build-$(date +%Y%m%d-%H%M%S).log"
    printf '\nBuilding. Log: %s\n' "$log"
    printf 'The first build takes 1-4 hours and will keep every core busy.\n\n'
    cd "$REPO_ROOT" || die "cannot enter $REPO_ROOT"

    # 'develop' installs a link to this source tree, so editing a .cpp/.cu/.py
    # file and re-running this script rebuilds only what changed.
    if time python setup.py develop 2>&1 | tee "$log"; then
        printf '\nBuild finished. Verify with: bash research/build.sh verify\n'
    else
        printf '\nBuild FAILED. The first error is the informative one:\n'
        grep -n -m5 -iE 'error:' "$log" || true
        printf '\nSee the troubleshooting table in research/BUILD.md.\n'
        return 1
    fi
}

# ================================================================== verify ===

run_verify() {
    # Run from outside the source tree: inside it, `import torch` resolves to the
    # source folder rather than the built extension and fails confusingly.
    # A source build reports 2.3.0a0+git<sha>, not a plain 2.3.0; verify.py accepts that.
    local script="$REPO_ROOT/research/verify.py"
    [ -r "$script" ] || die "missing $script"
    cd "$HOME" || cd / || true
    python "$script"
}

# ==================================================================== main ===

printf 'PyTorch v2.3.0 source build -- %s\n' "$MODE"
printf 'host %s, %s\n' "$(hostname 2>/dev/null)" "$(date -Is 2>/dev/null)"

load_env_file

if [ "$MODE" = "verify" ]; then
    run_verify
    exit $?
fi

detect_os
detect_cpu_ram
detect_disk
detect_gpu
detect_cuda
detect_compiler
detect_nvcc_smoke
detect_tooling
detect_python
detect_source
set_build_flags
print_plan

if [ "$FAIL" -gt 0 ]; then
    printf 'Resolve the %d [FAIL] item(s) above before building.\n' "$FAIL"
    exit 1
fi

if [ "$MODE" = "check" ]; then
    printf 'Ready to build: bash research/build.sh\n'
    exit 0
fi

if [ "${ASSUME_YES:-0}" != "1" ]; then
    if [ ! -t 0 ]; then
        die "not an interactive terminal; re-run with ASSUME_YES=1 to build unattended"
    fi
    # Drain input already buffered on stdin: pasting a block of commands leaves a
    # trailing newline behind, which would silently answer this prompt.
    while read -r -t 0 2>/dev/null; do read -r _ 2>/dev/null || break; done
    read -r -p "Proceed with the build? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) printf 'Aborted; nothing was built.\n'; exit 0 ;;
    esac
fi

run_build
