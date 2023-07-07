#! /bin/bash

CODAL_CACHE_DIR=$(mktemp -d)

CODAL_CACHE_DIR="$CODAL_CACHE_DIR" pytest "$@"

# Re-enable if/when we get lots of parallelizable tests
# CODAL_CACHE_DIR="$CODAL_CACHE_DIR" pytest -n auto -m "not serial" "$@" &&
# CODAL_CACHE_DIR="$CODAL_CACHE_DIR" pytest -m "serial" "$@"

EXIT_CODE=$?

rm -rf "$CODAL_CACHE_DIR"

exit $EXIT_CODE