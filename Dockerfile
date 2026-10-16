FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The core tool is standard-library only; install requirements.txt only if it
# lists any packages (comment and blank lines are ignored).
COPY requirements.txt .
RUN if grep -qvE '^\s*(#|$)' requirements.txt; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY main.py .
COPY ioc_analyzer/ ioc_analyzer/

# Mount reports at /app/raw_reports and collect results from /app/output.
CMD ["python", "main.py"]
