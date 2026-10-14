UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: all linux32 windows clean test lint format

all: linux32 windows

test:
	pytest
lint:
   	black --check . && isort --check-only . && flake8
format:
   	black . && isort .

linux32:
	docker build --platform linux/386 -t biblioteca-linux32 -f docker/Dockerfile.linux32 .
	mkdir -p dist/linux32
	docker run --rm --platform linux/386 -e HOST_UID=$(UID) -e HOST_GID=$(GID) -v "$(CURDIR)/dist/linux32:/out" biblioteca-linux32
	cd dist/linux32 && tar czf ../biblioteca-linux32.tar.gz biblioteca

windows:
	rm -rf dist/windows
	mkdir -p dist/windows
	docker run --rm -e HOST_UID=$(UID) -e HOST_GID=$(GID) -v "$(CURDIR):/src" tobix/pywine:3.9 bash /src/docker/build_windows.sh
	cd dist/windows && python3 -m zipfile -c ../biblioteca-windows.zip biblioteca

clean:
	rm -rf dist
