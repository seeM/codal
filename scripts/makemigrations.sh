#! /bin/bash

alembic -c codal/alembic/alembic.ini revision --autogenerate "$@"