#!/usr/bin/env python3

from __future__ import annotations
import os
from yandex_cloud_ml_sdk import YCloudML
from flask import Flask, request, Response
import json

FOLDER_ID = os.getenv("YC_FOLDER_ID", "not set")
API_KEY = os.getenv("YC_API_KEY", "not set")

print(f"folder_id={FOLDER_ID} api_key={API_KEY[:5]}...")

sdk = YCloudML(
    folder_id=FOLDER_ID,
    auth=API_KEY,
)

app = Flask(__name__)

def create_openai_response(text, role):
    r = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": role,
                    "content": text
                }
            }
        ]
    }
    return r

@app.route("/chat/completions", methods=["POST"])
def completion():
    body = request.get_json()
    messages = []
    for m in body["messages"]:
        messages.append({"role": m["role"], "text":m["content"]})

    def gen_chunks():
        l = 0
        for result in sdk.models.completions(f"gpt://{FOLDER_ID}/llama/latest").configure(temperature=0.5).run_stream(messages):
            for alternative in result:
                text = alternative.text[l:]
                l = len(alternative.text)
                response = create_openai_response(text, alternative.role)
                jresponse = json.dumps(response)
                yield 'data: %s\n\n' % jresponse

    return Response(gen_chunks(), mimetype="text/event-stream")
