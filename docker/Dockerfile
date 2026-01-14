FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.api"]
