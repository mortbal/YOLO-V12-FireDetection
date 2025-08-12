#!/usr/bin/env python3
"""
Fire Detection Dashboard Runner

Quick start script to launch the web dashboard
"""

import os
import sys

def main():
    # Change to the dashboard directory
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dashboard_dir)
    
    # Add backend directory to Python path
    backend_dir = os.path.join(dashboard_dir, 'backend')
    sys.path.insert(0, backend_dir)
    
    try:
        # Import and run the Flask app
        from backend.app import app, socketio
        print("🔥 Starting Fire Detection Dashboard...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("💡 Make sure to install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()