"""
GroomAI Public Deployment with Ngrok
Get an instant public URL accessible to everyone!
"""

import threading
import time
from pyngrok import ngrok
from app_enhanced import create_app

def run_flask():
    """Run Flask app"""
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main():
    print("=" * 60)
    print("  🚀 GroomAI Public Deployment")
    print("  Getting your public URL...")
    print("=" * 60)
    print()
    
    # Start Flask in background thread
    print("[1/3] Starting Flask server...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    time.sleep(5)
    print("✅ Flask server started on port 5000")
    
    # Start ngrok tunnel
    print("\n[2/3] Creating public tunnel...")
    try:
        public_url = ngrok.connect(5000, bind_tls=True)
        print("✅ Public tunnel created")
        
        # Display success message
        print("\n" + "=" * 60)
        print("  🎉 SUCCESS! Your GroomAI is now PUBLIC!")
        print("=" * 60)
        print()
        print(f"  🌐 Public URL: {public_url}")
        print()
        print("  📱 Share this link with anyone worldwide!")
        print("  🔒 HTTPS enabled automatically")
        print("  ⏱️  Link will remain active while this script runs")
        print()
        print("=" * 60)
        print()
        print("  📊 Usage Instructions:")
        print("  1. Copy the URL above")
        print("  2. Share it with anyone")
        print("  3. They can access your GroomAI instantly!")
        print()
        print("  ⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        print()
        
        # Keep running
        print("[3/3] Server is running...")
        print()
        print("📝 Server Logs:")
        print("-" * 60)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
            ngrok.disconnect(public_url)
            print("✅ Server stopped")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if port 5000 is already in use")
        print("2. Ensure you have internet connection")
        print("3. Try running again")

if __name__ == "__main__":
    main()
