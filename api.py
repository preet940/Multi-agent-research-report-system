"""
FastAPI service wrapping orchestrator.run(). Demonstrates the
background-job pattern discussed in the deployment notes: since a
report can take 30-60+ seconds, we don't hold the HTTP request open --
we return a job_id immediately and let the client poll for status.

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload

Then:
    curl -X POST http://localhost:8000/reports -H "Content-Type: application/json" -d '{"topic": "How is RAG used to reduce LLM hallucination?"}'
    -> {"job_id": "..."}
    curl http://localhost:8000/reports/<job_id>
    -> {"status": "running"} or {"status": "done", "result": {...}}
"""

import uuid
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from orchestrator import run

app = FastAPI(title="Research Report Agent API")

# In-memory job store. NOTE: this is fine for a single-instance demo,
# but breaks the moment you run >1 instance behind a load balancer --
# a real deployment needs a shared store (Redis, a database) instead,
# same "shared state across instances" issue mentioned for the vector
# cache in the scaling notes.
JOBS: dict = {}


class ReportRequest(BaseModel):
    topic: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"


def _run_job(job_id: str, topic: str):
    try:
        result = run(topic)
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/reports", response_model=JobStatus)
def create_report(req: ReportRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}

    # Run the (slow, blocking) pipeline in a background thread instead
    # of awaiting it directly in the request handler -- this is what
    # lets the API respond immediately instead of holding the connection
    # open for 30-60+ seconds.
    thread = threading.Thread(target=_run_job, args=(job_id, req.topic))
    thread.start()

    return {"job_id": job_id, "status": "running"}


@app.get("/reports/{job_id}")
def get_report(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/health")
def health():
    # A real health check -- load balancers and orchestrators (k8s, etc.)
    # poll this to know if an instance is alive and ready for traffic.
    return {"status": "ok"}