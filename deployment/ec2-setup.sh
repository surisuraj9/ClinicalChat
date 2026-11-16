#!/usr/bin/env bash
# One-shot setup of ClinicalChat on a fresh Ubuntu 22.04 / 24.04 EC2 instance (course Week 8).
# Usage (on the instance):
#   git clone https://github.com/surisuraj9/ClinicalChat.git && cd ClinicalChat
#   bash deployment/ec2-setup.sh            # installs, asks you to fill .env, starts the service on port 8000
# Re-run it after `git pull` to upgrade (it keeps your .env and data).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER="$(id -un)"
cd "$APP_DIR"

echo "==> System packages (OCR needs libgl / libglib)"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0

echo "==> uv + Python 3.11 virtual environment"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 --allow-existing .venv
uv pip install --python .venv/bin/python -r deployment/requirements.txt
uv pip install --python .venv/bin/python --no-deps .

echo "==> Secrets (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i 's/^PORT=.*/PORT=8000/' .env
  chmod 600 .env
  echo "Created .env from .env.example. Add OPENAI_API_KEY (and optionally LANGFUSE_*, TAVILY_API_KEY), and"
  echo "CHANGE DEMO_PASSWORD before exposing the app. Then re-run this script."
  ${EDITOR:-nano} .env || true
fi

echo "==> Parse the dataset and build the vector index (first run takes a few minutes)"
CLINICALCHAT_HOME="$APP_DIR" .venv/bin/clinicalchat ingest

echo "==> systemd service"
sed -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__USER__#$APP_USER#g" deployment/clinicalchat.service \
  | sudo tee /etc/systemd/system/clinicalchat.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable clinicalchat
sudo systemctl restart clinicalchat

echo "==> Waiting for the app to answer on :8000"
for _ in $(seq 1 60); do
  if curl -fs http://127.0.0.1:8000/api/health >/dev/null; then
    PUBLIC_IP="$(curl -fs -m 2 http://checkip.amazonaws.com || echo '<EC2-PUBLIC-IP>')"
    echo "ClinicalChat is up:  http://${PUBLIC_IP}:8000   (API docs: /docs)"
    exit 0
  fi
  sleep 5
done
echo "The service did not answer yet. Check: sudo journalctl -u clinicalchat -n 100 --no-pager" >&2
exit 1
