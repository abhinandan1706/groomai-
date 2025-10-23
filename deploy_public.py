"""
GroomAI Public Deployment using LocalTunnel
Creates a publicly accessible HTTPS URL for your skin analysis app
"""

import subprocess
import threading
import time
import os
import sys
from waitress import serve
from app import app

def run_flask():
    """Run Flask app with waitress"""
    print("🚀 Starting GroomAI server...")
    os.makedirs('uploads', exist_ok=True)
    serve(app, host='127.0.0.1', port=5000, threads=4)

def main():
    print("=" * 70)
    print("  🌐 GroomAI Public Deployment")
    print("  Creating your public HTTPS URL...")
    print("=" * 70)
    
    # Start Flask server in background
    print("\n[1/2] Starting Flask server...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    print("⏳ Waiting for server to start...")
    time.sleep(8)
    print("✅ Server is running on port 5000")
    
    # Start localtunnel
    print("\n[2/2] Creating public tunnel...")
    try:
        # Run localtunnel command
        result = subprocess.Popen([
            'lt', '--port', '5000', '--subdomain', 'groomai-skin-analysis'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a moment to start
        time.sleep(3)
        
        if result.poll() is None:  # Process is running
            print("✅ Public tunnel created!")
            print("\n" + "=" * 70)
            print("  🎉 SUCCESS! GroomAI is now PUBLIC!")
            print("=" * 70)
            print(f"\n  🌐 Public URL: https://groomai-skin-analysis.loca.lt")
            print(f"  📱 Alternative: Check terminal output below for exact URL")
            print("\n  🔒 HTTPS enabled automatically")
            print("  🌍 Accessible worldwide")
            print("  ⏱️  Link stays active while this runs")
            print("\n" + "=" * 70)
            print("\n  📋 How to use:")
            print("  1. Copy the URL above")
            print("  2. Share with anyone worldwide")
            print("  3. They can analyze their skin type instantly!")
            print(f"\n  🛑 Press Ctrl+C to stop")
            print("=" * 70)
            print("\n📝 Tunnel output:")
            print("-" * 70)
            
            try:
                # Show tunnel output
                while True:
                    output = result.stdout.readline()
                    if output:
                        print(output.strip())
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                result.terminate()
                print("✅ Server stopped")
        else:
            print("❌ Failed to create tunnel")
            stdout, stderr = result.communicate()
            if stderr:
                print(f"Error: {stderr}")
    
    except FileNotFoundError:
        print("❌ LocalTunnel not found. Installing...")
        subprocess.run(['npm', 'install', '-g', 'localtunnel'], check=True)
        print("✅ Installed. Please run the script again.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure port 5000 is available")
        print("2. Check your internet connection")
        print("3. Try running: npm install -g localtunnel")

if __name__ == "__main__":
    main()
