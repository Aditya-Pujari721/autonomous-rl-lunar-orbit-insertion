FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY lunar lunar
COPY viewer viewer
COPY model/policy.pt model/policy.pt
EXPOSE 8765
CMD ["python","-m","viewer.server","--host","0.0.0.0"]

