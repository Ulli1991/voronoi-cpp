#!/bin/bash
# Use a host Kokkos build configured with CMAKE_EXPORT_COMPILE_COMMANDS=ON.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$repo_root/build}"
if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "Missing compile_commands.json; configure a host build with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2
    exit 1
fi
for source in "$repo_root"/src/test/*.cpp "$repo_root"/src/bench/*.cpp; do
    clang-tidy -p "$build_dir" "$source"
done
