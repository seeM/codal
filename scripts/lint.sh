#! /bin/bash
black . &&
isort . &&
pyright
