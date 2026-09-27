#!/bin/bash
# Copy the adapter out of every finished checkpoint so each one can be
# evaluated after the trainer prunes it (save_total_limit).
#
#   archive_adapters.sh SAVES_DIR ARCHIVE_DIR [--once]
#
# A checkpoint is finished once trainer_state.json exists (written last).
# Loops every 60s until killed unless --once is given.
SAVES=$1
ARCHIVE=$2
ONCE=${3:-}
FILES="adapter_model.safetensors adapter_config.json trainer_state.json"

archive() {
    for ckpt in "$SAVES"/checkpoint-*; do
        [ -f "$ckpt/trainer_state.json" ] || continue
        dest="$ARCHIVE/$(basename "$ckpt")"
        [ -d "$dest" ] && continue
        mkdir -p "$ARCHIVE"
        tmp="$dest.tmp"
        rm -rf "$tmp" && mkdir -p "$tmp"
        for f in $FILES; do
            [ -f "$ckpt/$f" ] && cp "$ckpt/$f" "$tmp/"
        done
        mv "$tmp" "$dest"
        echo "$(date '+%F %T') archived $(basename "$ckpt")"
    done
}

if [ "$ONCE" = "--once" ]; then
    archive
    exit 0
fi
while true; do
    archive
    sleep 60
done
