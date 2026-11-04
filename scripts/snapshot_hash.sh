#!/bin/bash
# Hashes of the last snapshot in each run directory: U, x and volume (bits).
#   scripts/snapshot_hash.sh <run dir> [<run dir> ...]
# Needs h5dump (module hdf5-serial).
for dir in "$@"; do
    f="$(ls "$dir"/snapdir_*/*.hdf5 | tail -1)"
    sums=""
    for set in U x volume; do
        h5dump -d "/cells/$set" -b LE -o "$dir/$set.bin" "$f" > /dev/null
        sums+=" $set=$(sha256sum < "$dir/$set.bin" | cut -c1-16)"
    done
    echo "hash: $(basename "$dir")$sums"
done
