# ingest_api.py
from flask import Flask, request, jsonify
import threading
from model_example import update_model_metrics

app = Flask(__name__)

@app.route("/event", methods=["POST"])
def receive_event():
    """
    Receive an event and update model in real time.
    """
    data = request.get_json(force=True)
    

    result = update_model_metrics(data)
    
    return jsonify({"status": "ok", "metrics": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
