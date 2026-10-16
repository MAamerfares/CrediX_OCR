"""One-click launcher for the Egyptian Document OCR Web UI.

Usage:
    python run_ui.py
"""
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port

def open_browser(port: int):
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[+] Opening Web UI in your browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    import uvicorn

    port = find_available_port(8000)

    print("=" * 60)
    print("  EGYPTIAN DOCUMENT OCR & BANKING AI — WEB UI")
    print("=" * 60)
    if port != 8000:
        print(f"  Note: Port 8000 was busy. Switched automatically to port {port}!")
    print(f"  Server starting at: http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    # Launch browser automatically for the detected port
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start FastAPI server
    uvicorn.run("api:app", host="127.0.0.1", port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
