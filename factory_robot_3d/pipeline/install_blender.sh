#!/usr/bin/env bash
set -euo pipefail

VERSION="5.2.2"
ARCHIVE="blender-5.2.2-linux-x64.tar.xz"
BASE_URL="https://download.blender.org/release/Blender5.2"
URL="$BASE_URL/$ARCHIVE"
MANIFEST_URL="$BASE_URL/blender-5.2.2.sha256"
PINNED_SHA256="84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"

ROOT="${BLENDER_ROOT:-/kaggle/working/blender-runtime}"
CACHE="${BLENDER_CACHE:-/kaggle/working/blender-cache}"
INSTALL_DIR="$ROOT/blender-$VERSION-linux-x64"
BIN="$INSTALL_DIR/blender"

mkdir -p "$ROOT" "$CACHE"

if [[ -x "$BIN" ]]; then
  if "$BIN" --version | head -n1 | grep -Fxq "Blender $VERSION"; then
    echo "BLENDER_CACHE_HIT $BIN"
    printf '%s\n' "$BIN"
    exit 0
  fi
  rm -rf "$INSTALL_DIR"
fi

cd "$CACHE"
if [[ ! -s "$ARCHIVE" ]]; then
  curl --fail --location --retry 4 --retry-delay 3 "$URL" -o "$ARCHIVE"
fi

ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$PINNED_SHA256" ]]; then
  echo "Pinned SHA mismatch: expected $PINNED_SHA256 got $ACTUAL_SHA" >&2
  rm -f "$ARCHIVE"
  exit 1
fi

curl --fail --location --retry 4 "$MANIFEST_URL" -o "blender-$VERSION.sha256"
grep " $ARCHIVE$" "blender-$VERSION.sha256" > "blender-linux.sha256"
grep -Fq "$PINNED_SHA256" "blender-linux.sha256"
sha256sum -c "blender-linux.sha256"

rm -rf "$INSTALL_DIR"
tar -xJf "$ARCHIVE" -C "$ROOT"

if [[ ! -x "$BIN" ]]; then
  echo "Blender binary missing after extraction: $BIN" >&2
  exit 1
fi
"$BIN" --version | head -n1 | grep -Fx "Blender $VERSION"
printf '%s\n' "$BIN"
