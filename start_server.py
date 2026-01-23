"""
Quick startup script for RAG Debugger.
Run this to start the server after setting OPENAI_API_KEY.
"""
import os
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable not set!")
    print("   Set it with: $env:OPENAI_API_KEY='your-key-here'")
    print("   Or export OPENAI_API_KEY='your-key-here' on Linux/Mac")
    sys.exit(1)

try:
    from app import app
    print("[OK] App imports successfully")
    print("\nStarting RAG Debugger server...")
    print("   API will be available at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop the server\n")
except Exception as e:
    print(f"ERROR: Error importing app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
