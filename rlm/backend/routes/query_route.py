"""
RLM API routes.
"""

import json
import os
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from core import RLM_REPL
from config import settings
from utils.tracing import tracer

router = APIRouter(tags=["rlm"])


class QueryRequest(BaseModel):
    """Request model for RLM query."""

    query: str = Field(..., description="The question to answer about the context")
    context: Optional[str] = Field(None, description="Direct context string")
    context_file: Optional[str] = Field(None, description="Path to context file")
    max_iterations: int = Field(default=30, description="Maximum iterations for RLM")
    model: Optional[str] = Field(None, description="Model name to use")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What does this code do?",
                "context_file": "sample_code.py",
                "max_iterations": 30,
            }
        }


class QueryResponse(BaseModel):
    """Response model for RLM query."""

    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    cost_summary: Optional[dict] = None
    debug: Optional[dict] = None


class ContextFilesResponse(BaseModel):
    """Response model for listing context files."""

    files: list[str]
    base_path: str


CONTEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "contexts")


def _resolve_context(request: QueryRequest) -> str:
    """Resolve context text from inline payload or context file."""
    if request.context:
        return request.context

    if request.context_file:
        if os.path.isabs(request.context_file):
            file_path = request.context_file
        else:
            file_path = os.path.join(CONTEXT_DIR, request.context_file)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Context file not found: {request.context_file}",
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    raise HTTPException(
        status_code=400,
        detail="Either 'context' or 'context_file' must be provided",
    )


def _load_trace_rows(trace_file: Optional[str]) -> list[dict]:
    """Load trace rows from tracer JSONL file."""
    rows = []
    if not trace_file or not os.path.exists(trace_file):
        return rows

    with open(trace_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_debug_payload(
    *,
    request: QueryRequest,
    context: str,
    trace_file: Optional[str],
    engine_debug: Optional[dict],
) -> dict:
    """Build debug payload used by both sync and stream responses."""
    return {
        "trace_file": trace_file,
        "trace_rows": _load_trace_rows(trace_file),
        "context": {
            "source": "inline" if request.context else "context_file",
            "context_file": request.context_file,
            "query": request.query,
            "chars": len(context),
            "lines": context.count("\n") + 1 if context else 0,
            "preview": context[:4000] if context else "",
            "preview_truncated": bool(context and len(context) > 4000),
        },
        "engine": engine_debug or {},
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Run RLM query on the provided context.

    Either `context` (direct string) or `context_file` (path) must be provided.
    """
    context = _resolve_context(request)

    try:
        tracer.start_new_session()

        rlm = RLM_REPL(
            api_key=settings.llm.api_key,
            model=request.model or settings.llm.model_name,
            recursive_model=request.model or settings.llm.model_name,
            max_iterations=request.max_iterations,
        )

        result = rlm.completion(context=context, query=request.query)
        cost = rlm.cost_summary()
        debug_summary = rlm.debug_summary()

        trace_file = tracer.get_log_path()
        debug = _build_debug_payload(
            request=request,
            context=context,
            trace_file=trace_file,
            engine_debug=debug_summary,
        )

        if result is None:
            return QueryResponse(
                success=False,
                error="RLM reached max iterations without finding a final answer",
                cost_summary=cost,
                debug=debug,
            )

        return QueryResponse(
            success=True, answer=result, cost_summary=cost, debug=debug
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RLM error: {str(e)}")


@router.post("/query/stream")
async def run_query_stream(request: QueryRequest):
    """Stream query execution turn-by-turn using Server-Sent Events."""
    context = _resolve_context(request)
    context_meta = {
        "source": "inline" if request.context else "context_file",
        "context_file": request.context_file,
        "query": request.query,
        "chars": len(context),
        "lines": context.count("\n") + 1 if context else 0,
        "preview": context[:4000] if context else "",
        "preview_truncated": bool(context and len(context) > 4000),
    }

    tracer.start_new_session()
    rlm = RLM_REPL(
        api_key=settings.llm.api_key,
        model=request.model or settings.llm.model_name,
        recursive_model=request.model or settings.llm.model_name,
        max_iterations=request.max_iterations,
    )

    loop = asyncio.get_running_loop()
    completion_task = loop.run_in_executor(
        None, lambda: rlm.completion(context=context, query=request.query)
    )
    trace_file = tracer.get_log_path()

    async def event_stream():
        last_sent = 0

        started_payload = {
            "type": "started",
            "trace_file": trace_file,
            "context": context_meta,
        }
        yield f"data: {json.dumps(started_payload, ensure_ascii=False)}\n\n"

        try:
            while not completion_task.done():
                rows = _load_trace_rows(trace_file)
                while last_sent < len(rows):
                    row = rows[last_sent]
                    payload = {
                        "type": "turn",
                        "turn_index": last_sent,
                        "row": row,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    last_sent += 1

                await asyncio.sleep(0.35)

            rows = _load_trace_rows(trace_file)
            while last_sent < len(rows):
                row = rows[last_sent]
                payload = {
                    "type": "turn",
                    "turn_index": last_sent,
                    "row": row,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_sent += 1

            result = await completion_task
            cost = rlm.cost_summary()
            debug = _build_debug_payload(
                request=request,
                context=context,
                trace_file=trace_file,
                engine_debug=rlm.debug_summary(),
            )

            if result is None:
                complete_payload = {
                    "type": "completed",
                    "success": False,
                    "error": "RLM reached max iterations without finding a final answer",
                    "answer": None,
                    "cost_summary": cost,
                    "debug": debug,
                }
            else:
                complete_payload = {
                    "type": "completed",
                    "success": True,
                    "answer": result,
                    "cost_summary": cost,
                    "debug": debug,
                }

            yield f"data: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_payload = {
                "type": "error",
                "message": f"RLM error: {str(e)}",
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/contexts", response_model=ContextFilesResponse)
async def list_context_files():
    """List available context files in the contexts directory."""

    os.makedirs(CONTEXT_DIR, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(CONTEXT_DIR):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), CONTEXT_DIR)
            files.append(rel_path)

    return ContextFilesResponse(files=files, base_path=CONTEXT_DIR)


@router.post("/contexts/upload")
async def upload_context(filename: str, content: str):
    """Upload a context file."""

    os.makedirs(CONTEXT_DIR, exist_ok=True)

    file_path = os.path.join(CONTEXT_DIR, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(CONTEXT_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "path": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@router.get("/settings")
async def get_settings():
    """Get current RLM settings (without sensitive data)."""
    return {
        "model": settings.llm.model_name,
        "max_iterations": settings.rlm.max_iterations,
        "timeout": settings.llm.request_timeout,
    }
