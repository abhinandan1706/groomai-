"""
Start the base Flask app (app.py) and expose it publicly via ngrok.
This avoids the enhanced features by importing app from app.py directly.
Writes the public URL to public_url.txt.
"""
import threading
import time
from pyngrok import ngrok
from waitress import serve

# Import the base app without the new feature
from app import app

PORT = 5000


def run_server():
    # Production-grade WSGI server
    serve(app, host="0.0.0.0", port=PORT)


def main():
    print("Starting base app (app.py) on port", PORT, flush=True)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Give the server a moment to start
    time.sleep(4)

    print("Creating ngrok tunnel...", flush=True)
    tunnel = ngrok.connect(PORT, bind_tls=True)
    public_url = str(tunnel)

    # Output and persist the URL
    print("PUBLIC_URL=", public_url, flush=True)
    with open("public_url.txt", "w", encoding="utf-8") as f:
        f.write(public_url)

    print("Tunnel is live. Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        ngrok.disconnect(public_url)
        print("Shutting down...")


if __name__ == "__main__":
    main()

