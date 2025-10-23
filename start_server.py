from app import app
import os

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    print("🚀 Starting GroomAI server on http://127.0.0.1:5000")
    print("🌐 Ready for public tunnel!")
    app.run(host='127.0.0.1', port=5000, debug=False)
