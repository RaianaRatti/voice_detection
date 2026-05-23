# Notes:
#   1. HTTP requests: short-lived + unidirectional (server to client) --> Flask
#   2. WebSocket requests: long-lived + bidirectional --> Sock
#   3. JavaScript connects client to Websocket requests --> URL becomes /ws... (like /http...)

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
STATIC_DIR = Path(__file__).resolve().parent / "static"

from flask import Flask, send_from_directory, render_template
from flask_sock import Sock
import json
import threading
import time

from server.state import SharedState
from diarization.pipeline import run

# --------------------------------------------------------------------

app = Flask(__name__) # __name__ evaluates to name of current module (filename)
sock = Sock(app)

state = SharedState()

@app.route("/") # Flask automatically executes function following this and returns output to user's browser
def index():
    return render_template("index.html") # sends user to "index.html"


@app.route("/static/<path:filename>") # catch-all (if user visits "/static/css/main.css", filename = "css/main.css")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename) # sends user to 'filename'


@sock.route("/ws") # user connects to /ws, below function executes indefinitely
def websocket(ws): 
    while True:
        try:
            current_state = state.get()
            ws.send(json.dumps(current_state))
            time.sleep(0.5)

        except Exception:
            print("WebSocket disconnected")
            break


if __name__ == "__main__":

    # start diarization pipeline thread
    pipeline_thread = threading.Thread(
        target=run,
        args=(state,),
        daemon=True # background thread
    )

    pipeline_thread.start()

    # start Flask server - ONLY USE ON HOME NETWORK
    app.run(
        host="0.0.0.0", # allows other devices on your network connection to user your website
        port=5050,
        debug=True, # auto reload when new save + error line shows when error achieved
        use_reloader=False
    )