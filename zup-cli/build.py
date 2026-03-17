#!/usr/bin/env python3
"""
Build script for creating a portable Windows executable of Zup CLI.

Usage:
  python build.py
"""

import subprocess
import sys
import shutil
from pathlib import Path


def main():
    print("🔨 Building Zup CLI for Windows...\n")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Clean previous builds
    for path in ["build", "dist", "zup.spec"]:
        if Path(path).exists():
            print(f"🧹 Cleaning {path}...")
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
    
    # Build command
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable
        "--name=zup",                   # Output name
        "--console",                    # Console application
        "--clean",                      # Clean cache
        "--noconfirm",                  # Overwrite without asking
        "main.py"
    ]
    
    print(f"\n📦 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        exe_path = Path(__file__).parent / "dist" / "zup.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ Build successful!")
            print(f"📍 Executable: {exe_path}")
            print(f"📊 Size: {size_mb:.2f} MB")
            print(f"\n🚀 To distribute, copy 'dist/zup.exe' to any Windows machine.")
            print(f"   No Python installation required!")
        else:
            print("\n⚠️  Build completed but executable not found.")
            return 1
    else:
        print("\n❌ Build failed. Check the output above for errors.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
