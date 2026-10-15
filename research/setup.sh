#!/usr/bin/env bash
#
# One-time setup of a host for building PyTorch v2.3.0 (see build.sh).
#
#   bash research/setup.sh check      report what is present and what is missing; change nothing
#   bash research/setup.sh            install whatever is missing, then check again
#   bash research/setup.sh <step>     one step only: apt conda env cuda cudnn source
#
# Run it once per machine. Afterwards 'bash research/build.sh' builds, as often as
# needed; build.sh never installs anything.
#
# Steps, in order. Each one first looks at the real state of the machine (never at a
# "done" marker file), so re-running is always safe and re-does only what is missing:
#
#   apt     system packages: compilers, git, tmux, ccache. The only step that needs sudo.
#   conda   Miniforge (conda-forge only: no Anaconda terms-of-service prompt)
#   env     conda environment with Python 3.10 and the pins PyTorch 2.3.0 needs
#   cuda    CUDA Toolkit 12.1.1, installed under $HOME without root and without touching
#           the system CUDA or the driver
#   cudnn   cuDNN 8.9.7 for CUDA 12, unpacked into that toolkit
#   source  the git submodules of this checkout
#
# Settings (environment variables):
#   CONDA_ENV=pt230  CONDA_ROOT=$HOME/miniforge3  CUDA_PREFIX=$HOME/cuda-12.1
#   SETUP_DIR=$HOME/.cache/pt230-setup  ENV_FILE=$HOME/.pt230-env.sh
#   KEEP_DOWNLOADS=1   keep the CUDA and cuDNN archives after installing them
#   INSTALL_MAGMA=1    also install magma-cuda121 (optional; some torch.linalg CUDA ops)
#
# Only x86_64 Linux is supported: the download locations below are for that platform.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-pt230}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CUDA_PREFIX="${CUDA_PREFIX:-$HOME/cuda-12.1}"
SETUP_DIR="${SETUP_DIR:-$HOME/.cache/pt230-setup}"
ENV_FILE="${ENV_FILE:-$HOME/.pt230-env.sh}"

# Downloads are pinned to exact versions and verified against the publisher's own
# checksum, so a truncated or changed file stops the setup instead of failing an
# hour later. Sources: NVIDIA's md5sum.txt for CUDA 12.1.1, redistrib_8.9.7.json for
# cuDNN, and the GitHub release asset digest for Miniforge.
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh"
MINIFORGE_SHA256="281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"
CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run"
CUDA_RUN_MD5="2f0a4127bf797bf4eab0be2a547cb8d0"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
CUDNN_SHA256="475333625c7e42a7af3ca0b2f7506a106e30c93b1aa0081cd9c13efb6e21e3bb"

STEPS="apt conda env cuda cudnn source"
declare -A DEPENDS=( [env]=conda [cudnn]=cuda )

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ================================================================== helpers ===

# Run a command as root: directly if already root, else through passwordless sudo.
as_root() {
    if [ "$(id -u)" -eq 0 ]; then "$@"
    elif sudo -n true 2>/dev/null; then sudo "$@"
    else return 1
    fi
}

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

need_space() {   # need_space <GB> <why>
    local have; have="$(free_gb "$HOME")"
    [ -n "$have" ] && [ "$have" -ge "$1" ] \
        || die "$2 needs about $1 GB free under $HOME, found ${have:-unknown} GB"
}

# fetch_verified <url> <file> <md5|sha256> <expected>: resumable download that is
# thrown away and retried once when the checksum does not match.
fetch_verified() {
    local url="$1" file="$2" algo="$3" want="$4" got try
    mkdir -p "$(dirname "$file")" || return 1
    for try in 1 2; do
        wget -c --tries=5 --retry-connrefused --timeout=30 -O "$file" "$url" || { rm -f "$file"; continue; }
        got="$("${algo}sum" "$file" | awk '{print $1}')"
        if [ "$got" = "$want" ]; then return 0; fi
        say "  checksum mismatch (got $got, want $want); discarding $(basename "$file")"
        rm -f "$file"
    done
    return 1
}

# ================================================================== apt step ===

gcc_major() { command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1; }

# CUDA 12.1 rejects a host compiler newer than gcc 12. Ubuntu 24.04 ships gcc 13,
# so gcc-12 is added whenever the default compiler is newer than that.
apt_wanted() {
    local m; m="$(gcc_major)"
    printf '%s\n' build-essential git wget curl xz-utils ca-certificates tmux ccache
    if [ -n "$m" ] && [ "$m" -gt 12 ] 2>/dev/null; then printf '%s\n' gcc-12 g++-12; fi
}
apt_missing() {
    local p
    for p in $(apt_wanted); do
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || printf '%s\n' "$p"
    done
}
have_apt() { [ -z "$(apt_missing)" ]; }

do_apt() {
    command -v apt-get >/dev/null 2>&1 || { say "  not a Debian/Ubuntu host: install these with your package manager:"; apt_wanted | sed 's/^/    /'; return 1; }
    local pass missing
    # Two passes: the first installs the compiler, the second learns whether gcc-12 is needed.
    for pass in 1 2; do
        missing="$(apt_missing | tr '\n' ' ')"
        [ -z "$missing" ] && break
        say "  installing: $missing"
        if ! as_root apt-get update -qq || \
           ! as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y $missing; then
            say "  could not run apt as root. Ask an administrator, or run:"
            say "    sudo apt-get install -y $missing"
            return 1
        fi
    done
    command -v ccache >/dev/null 2>&1 && ccache -M 25G >/dev/null
    return 0
}

# ================================================================= conda steps ===

find_conda_sh() {
    local d base
    base="$(conda info --base 2>/dev/null || true)"
    for d in "$CONDA_ROOT" "$base" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge"; do
        if [ -n "$d" ] && [ -r "$d/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "$d/etc/profile.d/conda.sh"; return 0
        fi
    done
    return 1
}
conda_base() { local sh; sh="$(find_conda_sh)" || return 1; dirname "$(dirname "$(dirname "$sh")")"; }
env_prefix() { local b; b="$(conda_base)" || return 1; printf '%s/envs/%s\n' "$b" "$CONDA_ENV"; }

have_conda() { find_conda_sh >/dev/null 2>&1 && [ -x "$(conda_base)/bin/conda" ]; }

do_conda() {
    [ "$(uname -m)" = "x86_64" ] || { say "  Miniforge download is for x86_64 only"; return 1; }
    need_space 5 "Miniforge"
    local inst="$SETUP_DIR/miniforge.sh"
    fetch_verified "$MINIFORGE_URL" "$inst" sha256 "$MINIFORGE_SHA256" || return 1
    bash "$inst" -b -p "$CONDA_ROOT" || return 1
    rm -f "$inst"
}

# The environment is the one thing build.sh and Megatron-LM both import from, so its
# check is strict: Python 3.10, every module requirements.txt names, and the three
# pins (numpy<2, setuptools==69.5.1, cmake<4) that PyTorch 2.3.0 needs.
have_env() {
    local p py cm
    p="$(env_prefix 2>/dev/null)" || return 1
    py="$p/bin/python"
    [ -x "$py" ] || return 1
    "$py" - <<'PY' >/dev/null 2>&1 || return 1
import sys
assert sys.version_info[:2] == (3, 10), sys.version
import astunparse, expecttest, hypothesis, psutil, yaml, requests, typing_extensions
import sympy, filelock, networkx, jinja2, fsspec, packaging, optree, lark
import numpy, setuptools
assert numpy.__version__.startswith("1."), numpy.__version__
assert setuptools.__version__ == "69.5.1", setuptools.__version__
PY
    cm="$("$p/bin/cmake" --version 2>/dev/null | awk 'NR==1 {print $3}')"
    case "$cm" in 3.1[89].*|3.[2-9][0-9].*) ;; *) return 1 ;; esac
    [ -x "$p/bin/ninja" ] && [ -e "$p/lib/libmkl_intel_lp64.a" ] && [ -e "$p/include/mkl.h" ]
}

do_env() {
    local base prefix
    base="$(conda_base)" || { say "  conda is not installed: run the 'conda' step first"; return 1; }
    prefix="$(env_prefix)"
    need_space 8 "the conda environment"
    if [ ! -x "$prefix/bin/python" ]; then
        "$base/bin/conda" create -y -n "$CONDA_ENV" --override-channels -c conda-forge python=3.10 || return 1
    fi
    # Order matters: requirements.txt installs an unpinned setuptools and numpy, and the
    # pins that follow it put both back to the versions PyTorch 2.3.0 can build with.
    "$prefix/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt" || return 1
    "$base/bin/conda" install -y -p "$prefix" --override-channels -c conda-forge \
        "cmake<4" ninja mkl-static mkl-include || return 1
    "$prefix/bin/python" -m pip install "numpy<2" "setuptools==69.5.1" || return 1
    if [ "${INSTALL_MAGMA:-0}" = "1" ]; then
        "$base/bin/conda" install -y -p "$prefix" -c pytorch magma-cuda121 \
            || say "  magma-cuda121 failed to install; it is optional, continuing"
    fi
}

# ================================================================== cuda steps ===

cuda_ok() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version 2>/dev/null | grep -q 'release 12\.1,'; }

# Where CUDA 12.1 lives: this script's own prefix, or a system install.
cuda_home() {
    local d
    for d in "$CUDA_PREFIX" /usr/local/cuda-12.1; do
        if cuda_ok "$d"; then printf '%s\n' "$d"; return 0; fi
    done
    return 1
}
have_cuda() { cuda_home >/dev/null 2>&1; }

do_cuda() {
    [ "$(uname -m)" = "x86_64" ] || { say "  the CUDA runfile below is for x86_64 only"; return 1; }
    need_space 20 "CUDA 12.1 (a 4.3 GB download, about 5 GB installed)"
    local run="$SETUP_DIR/cuda_12.1.1_linux.run" log rc
    mkdir -p "$SETUP_DIR/tmp" || return 1
    fetch_verified "$CUDA_RUN_URL" "$run" md5 "$CUDA_RUN_MD5" || return 1
    # --toolkit only: no driver, so the driver on this machine is never touched.
    # --toolkitpath under $HOME: no root needed and /usr/local/cuda is left alone.
    # --override: skip the host-compiler check, which would reject gcc 13; build.sh
    # selects gcc-12 for the build itself. --tmpdir: /tmp can be too small to unpack.
    say "  installing the toolkit to $CUDA_PREFIX (a few minutes)"
    sh "$run" --silent --toolkit --toolkitpath="$CUDA_PREFIX" --override --no-man-page \
        --tmpdir="$SETUP_DIR/tmp"
    rc=$?
    rm -rf "$SETUP_DIR/tmp"
    if [ "$rc" -ne 0 ]; then
        say "  the CUDA installer exited with status $rc"
        for log in /var/log/cuda-installer.log "$HOME/cuda-installer.log" "$SETUP_DIR/cuda-installer.log"; do
            [ -r "$log" ] && { say "  last lines of $log:"; tail -n 15 "$log" | sed 's/^/    /'; break; }
        done
        return 1
    fi
    [ "${KEEP_DOWNLOADS:-0}" = "1" ] || rm -f "$run"
}

cudnn_header_ok() {   # cudnn_header_ok <cuda home>
    local h="$1/include/cudnn_version.h"
    [ -r "$h" ] || return 1
    [ "$(awk '/#define CUDNN_MAJOR/ {print $3}' "$h")" = "8" ] \
        && [ "$(awk '/#define CUDNN_MINOR/ {print $3}' "$h")" = "9" ] \
        && [ "$(awk '/#define CUDNN_PATCHLEVEL/ {print $3}' "$h")" = "7" ] \
        && [ -e "$1/lib64/libcudnn.so.8" ]
}
have_cudnn() { local h; h="$(cuda_home)" || return 1; cudnn_header_ok "$h"; }

do_cudnn() {
    local home tarball dir
    home="$(cuda_home)" || { say "  CUDA 12.1 is not installed: run the 'cuda' step first"; return 1; }
    need_space 8 "cuDNN"
    tarball="$SETUP_DIR/cudnn.tar.xz"
    dir="$SETUP_DIR/cudnn"
    fetch_verified "$CUDNN_URL" "$tarball" sha256 "$CUDNN_SHA256" || return 1
    rm -rf "$dir"; mkdir -p "$dir" || return 1
    tar -xf "$tarball" -C "$dir" --strip-components=1 || return 1
    # cp -P keeps the libcudnn.so -> .so.8 -> .so.8.9.7 symlink chain; a plain copy
    # would flatten it and break the loader.
    local cp_cmd=(cp -P)
    if [ ! -w "$home/include" ] || [ ! -w "$home/lib64" ]; then
        if [ "$(id -u)" -ne 0 ]; then cp_cmd=(sudo -n cp -P); fi
    fi
    "${cp_cmd[@]}" "$dir"/include/cudnn*.h "$home/include/" || return 1
    "${cp_cmd[@]}" "$dir"/lib/libcudnn* "$home/lib64/" || return 1
    rm -rf "$dir"
    [ "${KEEP_DOWNLOADS:-0}" = "1" ] || rm -f "$tarball"
}

# ================================================================= source step ===

have_source() {
    [ -f "$REPO_ROOT/setup.py" ] || return 1
    local total missing
    total="$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | wc -l)"
    missing="$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c '^-')"
    [ "${total:-0}" -gt 0 ] && [ "${missing:-1}" -eq 0 ]
}
do_source() {
    need_space 6 "the PyTorch submodules"
    git -C "$REPO_ROOT" submodule sync --recursive || return 1
    git -C "$REPO_ROOT" submodule update --init --recursive --jobs 8 || return 1
}

# ============================================================== env file step ===

# Records the choices made above so build.sh and a human shell get exactly the same
# CUDA and the same conda environment, without guessing again.
write_env_file() {
    local home sh
    home="$(cuda_home)" || return 1
    sh="$(find_conda_sh)" || return 1
    cat >"$ENV_FILE" <<EOF
# Written by research/setup.sh on $(date -Is). Re-run setup.sh instead of editing.
export CUDA_HOME="$home"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
if [ -z "\${CONDA_PREFIX:-}" ]; then
    . "$sh"
    conda activate "$CONDA_ENV"
fi
EOF
}

# ===================================================================== driver ===

run_check() {
    local s missing=0 label
    step "State of this host"
    for s in $STEPS; do
        if "have_$s"; then label="[have]   "; else label="[MISSING]"; missing=$((missing+1)); fi
        printf '  %s %s\n' "$label" "$s"
    done
    if [ -r "$ENV_FILE" ]; then say "  [have]    env file $ENV_FILE"
    else say "  [MISSING] env file $ENV_FILE (written at the end of a full setup)"; fi
    return "$missing"
}

FAILED=""
run_step() {
    local s="$1" dep
    step "$s"
    if "have_$s"; then say "  already in place"; return 0; fi
    dep="${DEPENDS[$s]:-}"
    if [ -n "$dep" ] && printf '%s' " $FAILED " | grep -q " $dep "; then
        say "  skipped: it needs the '$dep' step, which failed"
        FAILED="$FAILED $s"; return 1
    fi
    "do_$s" || { say "  FAILED: $s"; FAILED="$FAILED $s"; return 1; }
    # Never trust the installer's exit status alone: look at the machine again.
    if ! "have_$s"; then
        say "  FAILED: the $s step ran but its result is still not in place"
        FAILED="$FAILED $s"; return 1
    fi
    say "  done"
}

main() {
    local mode="${1:-all}" s log
    [ "$(uname -s)" = "Linux" ] || die "Linux only"
    case "$mode" in
        check|all|apt|conda|env|cuda|cudnn|source) ;;
        -h|--help|help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown step '$mode' (expected: check, all, $STEPS)" ;;
    esac

    if [ "$mode" = "check" ]; then run_check; exit $(( $? > 0 )); fi

    mkdir -p "$SETUP_DIR" || die "cannot create $SETUP_DIR"
    log="$SETUP_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$log") 2>&1
    printf 'PyTorch v2.3.0 host setup -- %s\nhost %s, %s, log %s\n' "$mode" "$(hostname)" "$(date -Is)" "$log"

    if [ "$mode" = "all" ]; then
        for s in $STEPS; do run_step "$s"; done
    else
        run_step "$mode"
    fi

    if [ -z "$FAILED" ] && [ "$mode" = "all" ]; then
        step "env file"
        if write_env_file; then say "  wrote $ENV_FILE"; else FAILED="$FAILED envfile"; fi
    fi

    step "Result"
    run_check
    if [ -n "$FAILED" ]; then
        printf '\nFailed steps:%s\nFix the cause (shown above), then re-run: bash research/setup.sh\n' "$FAILED"
        exit 1
    fi
    if [ "$mode" = "all" ]; then printf '\nSetup complete. Next: bash research/build.sh check\n'; fi
    exit 0
}

# Run main only when executed, so the functions can be sourced and tested one by one.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
