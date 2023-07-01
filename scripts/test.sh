# export CODAL_CACHE_DIR=$(mktemp -d)
export CODAL_CACHE_DIR=mytmpdir

pytest $@

rm -rf $CODAL_CACHE_DIR