"""Run: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"""
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import ChatRequest
from src.api.service import ChatService


def encode_sse(event):
    if event["event"] == "heartbeat":
        return ": heartbeat\n\n"
    return "event: " + event["event"] + "\ndata: " + json.dumps(event["data"], ensure_ascii=False) + "\n\n"


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        # Compile once at startup without creating models or calling APIs.
        app.state.chat = service if service is not None else ChatService()
        yield

    app = FastAPI(title="HW2 Research Assistant", lifespan=lifespan)

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    app.mount("/static", StaticFiles(directory=frontend), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(frontend / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok", "scope": "application only; providers not checked"}

    @app.post("/api/chat", response_class=StreamingResponse,
              responses={200: {"content": {"text/event-stream": {}}}})
    async def chat(body: ChatRequest):
        async def stream():
            events = app.state.chat.events(body.query)
            try:
                async for event in events:
                    yield encode_sse(event)
            finally:
                await events.aclose()
        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat/result")
    async def chat_result(body: ChatRequest):
        result = {"answer": "", "citations": [], "traces": [], "error": None}
        async for event in app.state.chat.events(body.query):
            data = event["data"]
            if event["event"] == "trace":
                result["traces"].append(data)
            elif event["event"] == "citation":
                result["citations"].append(data)
            elif event["event"] == "answer":
                result["answer"] = data["text"]
            elif event["event"] == "error":
                result["error"] = data
            elif event["event"] == "done":
                result.update(request_id=data["request_id"], status=data["status"], metrics=data["metrics"])
        return result

    return app


app = create_app()
