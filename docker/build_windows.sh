#!/bin/bash
set -e
cd /src
wine python -m pip install --upgrade pip
wine python -m pip install -r requirements-windows.txt
wine python /src/docker/freeze.py
wine python -m PyInstaller --onedir --noconfirm --noupx --name biblioteca \
    --collect-submodules sklearn --collect-submodules scipy --collect-data docx \
    --exclude-module matplotlib --exclude-module tkinter \
    --distpath /src/dist/windows --workpath /tmp/build --specpath /tmp \
    --collect-submodules segno \
    /src/servidor.py
chown -R "$HOST_UID:$HOST_GID" /src/dist /src/requirements-windows.lock
