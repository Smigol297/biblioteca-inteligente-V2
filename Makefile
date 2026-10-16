UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: all  clean test lint format

all: test format lint

test:
	pytest
lint:
	black --check . && isort --check-only . && flake8
format:
	black . && isort .

clean:
	rm -rf dist
