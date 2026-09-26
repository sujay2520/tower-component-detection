# 💻 VS Code Setup & 1-Command Execution Guide

## Quick Start in VS Code

### Method 1: The 1-Line Command
Open the integrated terminal in VS Code (`Ctrl + ~`) and run:
```bash
python run.py
```

### Method 2: Press F5 (Launch Configuration)
1. Open the project folder in VS Code.
2. Press **`F5`** on your keyboard (or click **Run ➔ Start Debugging**).
3. VS Code automatically uses `.vscode/launch.json` to start the backend and open your browser at `http://localhost:5000`.

### Method 3: Press Ctrl+Shift+B (Build Task)
1. Press **`Ctrl + Shift + B`** in VS Code.
2. Select **`Run Tower Detection System`**.

### Method 4: Double-Click on Windows
- Double-click `start.bat` in the project root folder.

## System Endpoints & Ports

| Endpoint | Purpose | Method |
| :--- | :--- | :--- |
| `http://localhost:5000/` | Main Interactive Dashboard | GET |
| `http://localhost:5000/health` | Backend & Model Status | GET |
| `http://localhost:5000/upload` | Image Upload Endpoint | POST |
| `http://localhost:5000/execute` | Detection & Structural Analysis | POST |
| `http://localhost:5000/metrics` | Evaluation Metrics JSON | GET |
| `http://localhost:5000/graphs` | Training & Validation Plots | GET |
