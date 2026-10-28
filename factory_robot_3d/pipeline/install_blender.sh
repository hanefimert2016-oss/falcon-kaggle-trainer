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

CURL_ARGS=(
  --fail
  --location
  --show-error
  --retry 8
  --retry-all-errors
  --retry-delay 5
  --connect-timeout 20
  --max-time 1800
)

download_to_file() {
  local url="$1"
  local dest="$2"
  local part="${dest}.part"

  rm -f "$part"
  echo "BLENDER_INSTALL_STAGE download url=$url dest=$dest" >&2
  curl "${CURL_ARGS[@]}" "$url" -o "$part"
  if [[ ! -s "$part" ]]; then
    echo "Downloaded file is empty: $part" >&2
    rm -f "$part"
    return 1
  fi
  mv -f "$part" "$dest"
}

first_version_line() {
  local output first
  output="$("$BIN" --version 2>&1)"
  first="${output%%$'\n'*}"
  printf '%s\n' "$first"
}

if [[ -x "$BIN" ]]; then
  CACHED_VERSION="$(first_version_line || true)"
  if [[ "$CACHED_VERSION" == "Blender $VERSION"* ]]; then
    echo "BLENDER_CACHE_HIT $BIN"
    printf '%s\n' "$BIN"
    exit 0
  fi
  echo "BLENDER_INSTALL_STAGE stale_cache version=$CACHED_VERSION" >&2
  rm -rf "$INSTALL_DIR"
fi

cd "$CACHE"

if [[ -s "$ARCHIVE" ]]; then
  ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA" != "$PINNED_SHA256" ]]; then
    echo "BLENDER_INSTALL_STAGE stale_archive expected=$PINNED_SHA256 got=$ACTUAL_SHA" >&2
    rm -f "$ARCHIVE"
  fi
fi

if [[ ! -s "$ARCHIVE" ]]; then
  download_to_file "$URL" "$ARCHIVE"
fi

echo "BLENDER_INSTALL_STAGE verify_pinned_sha" >&2
ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$PINNED_SHA256" ]]; then
  echo "Pinned SHA mismatch: expected $PINNED_SHA256 got $ACTUAL_SHA" >&2
  rm -f "$ARCHIVE"
  exit 1
fi
printf '%s  %s\n' "$PINNED_SHA256" "$ARCHIVE" | sha256sum -c -

echo "BLENDER_INSTALL_STAGE verify_official_manifest" >&2
MANIFEST_FILE="blender-$VERSION.sha256"
download_to_file "$MANIFEST_URL" "$MANIFEST_FILE"
grep -F "$ARCHIVE" "$MANIFEST_FILE" > "blender-linux.sha256"
if [[ ! -s "blender-linux.sha256" ]]; then
  echo "Official Blender manifest does not contain $ARCHIVE" >&2
  exit 1
fi
if ! grep -Fq "$PINNED_SHA256" "blender-linux.sha256"; then
  echo "Official Blender manifest does not contain pinned SHA $PINNED_SHA256" >&2
  cat "blender-linux.sha256" >&2
  exit 1
fi
sha256sum -c "blender-linux.sha256"

echo "BLENDER_INSTALL_STAGE extract root=$ROOT" >&2
rm -rf "$INSTALL_DIR"
tar -xJf "$ARCHIVE" -C "$ROOT"

if [[ ! -x "$BIN" ]]; then
  echo "Blender binary missing after extraction: $BIN" >&2
  exit 1
fi

echo "BLENDER_INSTALL_STAGE verify_binary" >&2
VERSION_LINE="$(first_version_line)"
if [[ "$VERSION_LINE" != "Blender $VERSION"* ]]; then
  echo "Unexpected Blender version: $VERSION_LINE" >&2
  exit 1
fi

echo "BLENDER_INSTALL_OK version=$VERSION path=$BIN" >&2
printf '%s\n' "$BIN"
