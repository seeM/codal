#! /bin/bash

CODAL_CACHE_DIR=$(mktemp -d)

CODAL_CACHE_DIR="$CODAL_CACHE_DIR" pytest "$@" &&

rm -rf "$CODAL_CACHE_DIR"
