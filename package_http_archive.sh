#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob globstar

usage() {
  cat <<'EOF'
Usage:
  package_http_archive.sh [options]

Options:
  --project-dir <path>   Project root directory (default: script directory)
  --build-dir <path>     Build directory containing .so/.a (default: <project-dir>/build)
  --headers-dir <path>   Header source directory (default: <project-dir>/cpp)
  --output-dir <path>    Output directory for bundle/tar.gz (default: <project-dir>)
  --arch <str>           Architecture tag in default bundle name (default: x86_64)
  --bundle-prefix <str>  Output bundle prefix (default: tensorrt-edge-llm-<arch>)
  --include-example-utils
                         Include libexampleUtils.a from examples/utils (default: disabled)
  --keep-verify-dir      Keep temporary extraction directory used for verification
  --help                 Show this help

Package layout:
  <bundle>/
    include/tensorrt-edge-llm/**/*
    lib/*

Behavior:
  - Copies header-like files: .h/.hpp/.hh/.cuh from headers-dir
  - Copies all .so/.so.* and .a from build-dir into lib/
  - Excludes libexampleUtils.a by default (enable with --include-example-utils)
  - Preserves .so symlinks
  - Creates <bundle>.tar.gz
  - Verifies symlink integrity after extraction
EOF
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s/%s\n" "$(pwd)" "$path"
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$script_dir"
build_dir=""
headers_dir=""
output_dir=""
arch="x86_64"
bundle_prefix=""
include_example_utils="0"
keep_verify_dir="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      project_dir="$(abs_path "$2")"
      shift 2
      ;;
    --build-dir)
      build_dir="$(abs_path "$2")"
      shift 2
      ;;
    --headers-dir)
      headers_dir="$(abs_path "$2")"
      shift 2
      ;;
    --output-dir)
      output_dir="$(abs_path "$2")"
      shift 2
      ;;
    --arch)
      arch="$2"
      shift 2
      ;;
    --bundle-prefix)
      bundle_prefix="$2"
      shift 2
      ;;
    --include-example-utils)
      include_example_utils="1"
      shift
      ;;
    --keep-verify-dir)
      keep_verify_dir="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

build_dir="${build_dir:-$project_dir/build}"
headers_dir="${headers_dir:-$project_dir/cpp}"
output_dir="${output_dir:-$project_dir}"

if [[ -z "$arch" ]]; then
  echo "Architecture cannot be empty." >&2
  exit 1
fi
if [[ ! "$arch" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid architecture: $arch (allowed: letters, numbers, dot, underscore, hyphen)" >&2
  exit 1
fi

bundle_prefix="${bundle_prefix:-tensorrt-edge-llm-${arch}}"

if [[ ! -d "$project_dir" ]]; then
  echo "Project directory not found: $project_dir" >&2
  exit 1
fi
if [[ ! -d "$build_dir" ]]; then
  echo "Build directory not found: $build_dir" >&2
  exit 1
fi
if [[ ! -d "$headers_dir" ]]; then
  echo "Headers directory not found: $headers_dir" >&2
  exit 1
fi
if [[ ! -d "$output_dir" ]]; then
  echo "Output directory not found: $output_dir" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
bundle_name="${bundle_prefix}_${timestamp}"
bundle_dir="${output_dir}/${bundle_name}"
archive_path="${output_dir}/${bundle_name}.tar.gz"
verify_dir="${output_dir}/.verify_${bundle_name}"

mkdir -p "${bundle_dir}/include/tensorrt-edge-llm" "${bundle_dir}/lib"

declare -a so_candidates=()
declare -a static_candidates=()
declare -A seen_lib_basename=()

for f in "${build_dir}"/**/*.so "${build_dir}"/**/*.so.*; do
  [[ -e "$f" ]] || continue
  so_candidates+=("$f")
done
for f in "${build_dir}"/**/*.a; do
  [[ -e "$f" ]] || continue
  if [[ "$include_example_utils" != "1" ]] && [[ "$(basename "$f")" == "libexampleUtils.a" ]]; then
    continue
  fi
  static_candidates+=("$f")
done

if [[ ${#so_candidates[@]} -eq 0 ]]; then
  echo "No shared library files found in: $build_dir" >&2
  exit 1
fi

for f in "${so_candidates[@]}" "${static_candidates[@]}"; do
  base_name="$(basename "$f")"
  if [[ -n "${seen_lib_basename[$base_name]:-}" ]]; then
    echo "Duplicate library basename detected: $base_name" >&2
    echo "  Existing: ${seen_lib_basename[$base_name]}" >&2
    echo "  Incoming: $f" >&2
    exit 1
  fi
  seen_lib_basename["$base_name"]="$f"
  cp -a "$f" "${bundle_dir}/lib/"
done

rsync -a \
  --prune-empty-dirs \
  --include '*/' \
  --include '*.h' \
  --include '*.hpp' \
  --include '*.hh' \
  --include '*.cuh' \
  --exclude '*' \
  "${headers_dir}/" \
  "${bundle_dir}/include/tensorrt-edge-llm/"

if [[ ! -f "${bundle_dir}/include/tensorrt-edge-llm/runtime/llmInferenceRuntime.h" ]]; then
  echo "Expected header missing: include/tensorrt-edge-llm/runtime/llmInferenceRuntime.h" >&2
  exit 1
fi

tar -czf "$archive_path" -C "$output_dir" "$bundle_name"

rm -rf "$verify_dir"
mkdir -p "$verify_dir"
tar -xzf "$archive_path" -C "$verify_dir"

verify_lib_dir="${verify_dir}/${bundle_name}/lib"
for lib_entry in "$verify_lib_dir"/*; do
  [[ -e "$lib_entry" ]] || continue
  if [[ -L "$lib_entry" ]]; then
    target="$(readlink "$lib_entry")"
    if [[ ! -e "${verify_lib_dir}/${target}" ]]; then
      echo "Broken symlink after extraction: $lib_entry -> $target" >&2
      exit 1
    fi
  fi
done

header_count=0
for f in \
  "${bundle_dir}/include/tensorrt-edge-llm"/**/*.h \
  "${bundle_dir}/include/tensorrt-edge-llm"/**/*.hpp \
  "${bundle_dir}/include/tensorrt-edge-llm"/**/*.hh \
  "${bundle_dir}/include/tensorrt-edge-llm"/**/*.cuh; do
  [[ -e "$f" ]] || continue
  header_count=$((header_count + 1))
done

lib_count=0
for f in "${bundle_dir}/lib"/*; do
  [[ -e "$f" ]] || continue
  lib_count=$((lib_count + 1))
done

sha256="$(sha256sum "$archive_path" | awk '{print $1}')"
archive_size="$(du -h "$archive_path" | awk '{print $1}')"

echo "BUNDLE_DIR=${bundle_dir}"
echo "ARCHIVE=${archive_path}"
echo "SHA256=${sha256}"
echo "SIZE=${archive_size}"
echo "HEADER_COUNT=${header_count}"
echo "LIB_COUNT=${lib_count}"
echo "INCLUDE_ROOT=${bundle_dir}/include"
echo "CHECK_INCLUDE_EXAMPLE=#include \"tensorrt-edge-llm/runtime/llmInferenceRuntime.h\""

if [[ "$keep_verify_dir" == "1" ]]; then
  echo "VERIFY_DIR_KEPT=${verify_dir}"
else
  rm -rf "$verify_dir"
fi
