#!/usr/bin/env bash
set -euo pipefail

PATCH="$1"

if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "[patch] already applied"
    exit 0
fi

if git apply --check "$PATCH" >/dev/null 2>&1; then
    echo "[patch] applying $PATCH"
    git apply "$PATCH"
    exit 0
fi

echo "[patch] ERROR: patch neither applies nor appears already applied"
git apply --check "$PATCH"
exit 1
