"""
serve_frontend.py — Multi-threaded robust HTTP server for the Frontend Dashboard.
==================================================================================
Runs on port 8080 with ThreadingHTTPServer and SO_REUSEADDR.
"""

import os
import sys
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = 8080
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[Frontend 8080] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


class ResilientServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if not os.path.exists(FRONTEND_DIR):
        print(f"Error: Frontend directory not found at {FRONTEND_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("  FRONTEND DASHBOARD HTTP SERVER")
    print("=" * 60)
    print(f"  Directory : {FRONTEND_DIR}")
    print(f"  Port      : {PORT}")
    print(f"  URL       : http://localhost:{PORT}")
    print("=" * 60)

    server = ResilientServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server...")
        server.server_close()


if __name__ == "__main__":
    main()
