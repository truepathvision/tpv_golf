# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect onnxruntime and insightface data files
datas = []
datas += collect_data_files("onnxruntime")
datas += collect_data_files("insightface")

hidden_imports = collect_submodules("onnxruntime") + collect_submodules("insightface")

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=[],
    datas=datas + [("assets/icon.png", "assets")],
    hiddenimports=hidden_imports + [
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "cv2",
        "numpy",
        "faiss",
        "PIL",
        "PIL.Image",
        "PIL.ImageOps",
        "PIL.ExifTags",
        "huggingface_hub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TPV_Golf",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TPV_Golf",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="TPV_Golf.app",
        icon="assets/icon.png",
        bundle_identifier="com.tpv.golf",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "1.0.0",
        },
    )
