from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import tempfile
import wave
import os
import asyncio

from interview import evaluate_interview_answers
from llm import llm
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

# ✅ Whisper model - "base" is good, "tiny" is faster for CPUs
model = WhisperModel("base", device="cpu", compute_type="int8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WS connected ✅")
    audio_buffer = []

    try:
        while True:
            data = await ws.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)
            audio_buffer.extend(chunk)

            # Process every ~3 seconds
            if len(audio_buffer) > 16000 * 3:
                # Capture current buffer and clear it immediately
                current_audio = np.array(audio_buffer)
                audio_buffer = []

                # 🔥 FIX: Run transcription in a separate thread to prevent Ping Timeout
                text = await asyncio.to_thread(run_transcription, current_audio)
                
                if text.strip():
                    print(f"Sending to frontend: {text}") # Check your terminal for this!
                    await ws.send_text(text.strip())

    except WebSocketDisconnect:
        print("Client disconnected 👋")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Helper function for the thread
def run_transcription(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        with wave.open(temp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())
        
    # Transcription happens here, off the main event loop
    segments, _ = model.transcribe(temp.name, beam_size=5)
    text = "".join([seg.text for seg in segments])
    
    try:
        os.unlink(temp.name)
    except:
        pass
    return text

class RoleInput(BaseModel):
    role: str

@app.post("/generate-questions")
async def generate_questions(data: RoleInput):
    prompt = f"Generate 5 technical interview questions for a {data.role}. Return ONLY a JSON list of strings."
    response = llm.invoke(prompt)
    try:
        # Handling potential markdown backticks from LLM
        content = response.content.replace("```json", "").replace("```", "").strip()
        questions = json.loads(content)
        return {"questions": questions}
    except:
        return {"questions": [q.strip() for q in response.content.split("\n") if len(q) > 5][:5]}

@app.post("/evaluate-interview")
async def evaluate_interview(qalist: str = Form(...)):
    # Since we sent JSON string via FormData in React
    data = json.loads(qalist)
    # Convert list of dicts to objects for the helper function
    results = await evaluate_interview_answers(data)
    return results