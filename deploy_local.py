"""
Simple Local Network Deployment for GroomAI
Access your app from any device on your local network!
"""

import os
from waitress import serve
from app import app

def main():
    print("=" * 60)
    print("  🚀 GroomAI Local Network Deployment")
    print("=" * 60)
    
    # Make sure upload directory exists
    os.makedirs('uploads', exist_ok=True)
    
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 5000
    
    print(f"\n🌐 Starting server on all network interfaces...")
    print(f"📱 Access from any device on your network:")
    print(f"   • Local: http://localhost:{PORT}")
    print(f"   • Network: http://[YOUR_IP]:{PORT}")
    print(f"\n💡 To find your IP address, run: ipconfig")
    print(f"🛑 Press Ctrl+C to stop\n")
    print("=" * 60)
    
    try:
        # Use production WSGI server
        serve(app, host=HOST, port=PORT, threads=4)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gracefully...")
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
