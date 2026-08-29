# Base image: slim = smaller image, faster pulls/deploys, fewer
# unnecessary packages (smaller attack surface too).
FROM python:3.11-slim

WORKDIR /app

# Copy requirements FIRST, before the rest of the code, and install
# them in a separate layer. Docker caches layers -- if only your .py
# files change (not requirements.txt), this pip install layer is
# reused from cache instead of re-running, which makes rebuilds much
# faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Now copy the actual application code
COPY . .

# Document which port the app listens on (informational -- doesn't
# actually publish the port, that happens at `docker run -p`)
EXPOSE 8000

# Environment variables like GROQ_API_KEY, TAVILY_API_KEY are NOT baked
# into the image (never hardcode secrets into a Dockerfile/image layer --
# anyone with the image can extract them). Pass them at runtime instead:
#   docker run -e GROQ_API_KEY=xxx -e TAVILY_API_KEY=xxx ...

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]