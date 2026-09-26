"""
=============================================================================
Tower Component Detection & Structural Assessment System
=============================================================================
Single-command launcher for VS Code and terminal:
    python run.py

This automatically:
  1. Verifies the trained YOLOv8 model weights exist.
  2. Serves the full application (Backend API + Interactive Frontend) on:
     http://localhost:5000
  3. Automatically opens your default web browser to the dashboard.
=============================================================================
"""

import os
import sys
import time
import webbrowser
from threading import Timer
from pathlib import Path

# Ensure UTF-8 output on Windows consoles if supported
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Set working directory to project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

def open_dashboard():
    time.sleep(1.2)
    url = "http://localhost:5000"
    print("\n" + "=" * 60)
    print("   Tower Component Detection System is READY!")
    print(f"   Dashboard URL: {url}")
    print("=" * 60 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    print("\n" + "=" * 60)
    print("  TOWER COMPONENT DETECTION & CLASSIFICATION SYSTEM")
    print("=" * 60)

    # Verify model weights
    weights_path = PROJECT_ROOT / "runs" / "tower_final" / "weights" / "best.pt"
    if weights_path.exists():
        print(f"[OK] Trained weights found: {weights_path}")
    else:
        print(f"[WARN] Trained weights not found at {weights_path}, will attempt fallback search.")

    # Schedule browser launch
    Timer(1.5, open_dashboard).start()

    # Import and run Flask app
    from backend.app import app
    print("[INFO] Starting server on http://localhost:5000 (Press Ctrl+C to stop)...")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()
