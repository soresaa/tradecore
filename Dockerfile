FROM python:3.14-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 DATA_DIR=/tmp/tradecore-data PORT=7860
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-m", "cloud.server"]
