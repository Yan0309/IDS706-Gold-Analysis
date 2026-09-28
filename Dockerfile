FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY hw3_analysis.py ./
COPY data/ ./data/
RUN mkdir -p /app/figures
CMD ["python", "hw3_analysis.py"]