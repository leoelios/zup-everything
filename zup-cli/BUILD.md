# Building Zup CLI for Windows

This guide explains how to create a portable, standalone executable of Zup CLI for Windows.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Quick Start

### 1. Install Build Dependencies

```bash
cd zup-cli
pip install -r requirements-build.txt
```

### 2. Build the Executable

```bash
python build.py
```

This will:
- Install PyInstaller if needed
- Clean previous builds
- Create a single `zup.exe` file in the `dist/` folder
- Bundle all dependencies and required assets (including `assets/icon.txt`) so no Python is required on the target machine

### 3. Distribute

The executable is located at:
```
zup-cli/dist/zup.exe
```

**Copy this file to any Windows machine** — it will run without Python installed.

## Usage on Target Machine

```bash
# Configure credentials (first time)
zup.exe --config

# Interactive mode
zup.exe

# One-shot command
zup.exe "your prompt here"

# Debug mode
zup.exe --debug
```

## Manual Build (Advanced)

If you need custom PyInstaller options:

```bash
pyinstaller --onefile --name=zup --console main.py
```

### Common Options

- `--onefile` — Single executable (vs. folder with dependencies)
- `--windowed` — No console window (for GUI apps)
- `--icon=icon.ico` — Custom icon
- `--add-data="src;dest"` — Include additional files
- `--hidden-import=module` — Force include a module

## Known Issues & Fixes (from real builds)

### 1. UnicodeEncodeError with emoji characters on Windows

**Symptom:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f528'
```

**Fix:** Run the build script with UTF-8 mode enabled:
```bash
PYTHONUTF8=1 python build.py
```

Or set the environment variable permanently:
```bash
set PYTHONUTF8=1
python build.py
```

---

### 2. `pathlib` backport incompatible with PyInstaller

**Symptom:**
```
ERROR: The 'pathlib' package is an obsolete backport of a standard library package
and is incompatible with PyInstaller.
```

**Fix:** Uninstall the obsolete `pathlib` backport before building:
```bash
pip uninstall pathlib -y
python build.py
```

> This package is a backport that conflicts with Python 3's built-in `pathlib`. Safe to remove on Python 3.4+.

---

## Troubleshooting

### "Module not found" errors

If the built executable fails with import errors, add hidden imports:

```bash
pyinstaller --onefile --hidden-import=missing_module main.py
```

### Large executable size

The executable includes the Python interpreter and all dependencies (~35-40 MB is normal for this project).

To reduce size:
- Use `--exclude-module` to remove unused packages
- Consider UPX compression: `pyinstaller --onefile --upx-dir=/path/to/upx main.py`

### Antivirus false positives

Some antivirus software flags PyInstaller executables. This is a known issue.

Solutions:
- Code-sign the executable (requires a certificate)
- Submit to antivirus vendors as false positive
- Distribute source code instead

## Alternative: Python Virtual Environment

If you prefer not to use PyInstaller, you can distribute a portable Python environment:

```bash
# Create virtual environment
python -m venv zup-env

# Activate and install
zup-env\Scripts\activate
pip install -r requirements.txt

# Distribute the entire zup-env folder
# Users run: zup-env\Scripts\python.exe main.py
```

## CI/CD Integration

For automated builds, add to your workflow:

```yaml
# .github/workflows/build.yml
name: Build Windows Executable

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: |
          cd zup-cli
          pip install -r requirements-build.txt
          python build.py
      - uses: actions/upload-artifact@v3
        with:
          name: zup-windows
          path: zup-cli/dist/zup.exe
```

## Distribution Checklist

- [ ] Build executable with `python build.py`
- [ ] Test on a clean Windows machine (no Python installed)
- [ ] Verify all features work (auth, file operations, etc.)
- [ ] Check executable size is reasonable
- [ ] Include README with usage instructions
- [ ] Consider code-signing for production releases
