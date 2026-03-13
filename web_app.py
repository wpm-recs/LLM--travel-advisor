"""Simple web server for Travel Advisor RAG (no extra web framework required)."""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from main import TravelRAGSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
SYSTEM_LOCK = Lock()
RAG_SYSTEM: TravelRAGSystem | None = None


def _consume_answer(result: Any) -> str:
    """Normalize string or streaming generator into a final text answer."""
    if isinstance(result, str):
        return result

    if result is None:
        return ""

    try:
        return "".join(part for part in result)
    except TypeError:
        return str(result)


def initialize_rag_system() -> TravelRAGSystem:
    """Initialize and cache the RAG system exactly once."""
    global RAG_SYSTEM

    with SYSTEM_LOCK:
        if RAG_SYSTEM is not None:
            return RAG_SYSTEM

        logger.info("Initializing Travel RAG system...")
        system = TravelRAGSystem()
        system.initialize_system()
        system.build_knowledge_base()
        RAG_SYSTEM = system
        logger.info("Travel RAG system is ready.")
        return RAG_SYSTEM


class TravelAdvisorHandler(BaseHTTPRequestHandler):
    server_version = "TravelAdvisorHTTP/1.0"

    def _send_json(self, data: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
            return

        if self.path == "/styles.css":
            self._serve_file(FRONTEND_DIR / "styles.css", "text/css; charset=utf-8")
            return

        if self.path == "/app.js":
            self._serve_file(FRONTEND_DIR / "app.js", "application/javascript; charset=utf-8")
            return

        if self.path == "/api/health":
            ready = RAG_SYSTEM is not None
            self._send_json({"status": "ok" if ready else "initializing", "ready": ready})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_payload = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload."}, HTTPStatus.BAD_REQUEST)
            return

        question = (payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Question is required."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            system = initialize_rag_system()
            result = system.ask_question(question, stream=True)
            answer = _consume_answer(result)
            self._send_json({"answer": answer, "question": question})
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            logger.exception("Failed to answer question")
            self._send_json(
                {"error": f"Failed to answer question: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def run_server() -> None:
    host = os.getenv("TRAVEL_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("TRAVEL_WEB_PORT", "8080"))

    logger.info("Bootstrapping RAG system. This may take a while on first launch...")
    initialize_rag_system()

    httpd = ThreadingHTTPServer((host, port), TravelAdvisorHandler)
    logger.info("Travel Advisor web server running at http://%s:%s", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server()
