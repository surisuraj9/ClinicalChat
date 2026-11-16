# Deploying ClinicalChat on AWS EC2 (course Week 8)

The app is one Python process: FastAPI serves the REST API (`/api/...`, Swagger at `/docs`) and the Gradio chat UI
(at `/`) on **port 8000**. systemd keeps it running and restarts it after a crash or reboot.

| File | What it is |
|------|------------|
| `ec2-setup.sh` | One-shot install / upgrade on Ubuntu: system packages, Python 3.11 venv, pinned dependencies, `.env`, dataset ingest, systemd service, health check |
| `clinicalchat.service` | systemd unit template (the script fills in the folder and user) |
| `requirements.txt` | Exact package versions verified locally |
| `../Dockerfile` | Docker alternative (same port 8000) |

## 1. Launch the instance

1. EC2 → **Launch instance** → *Ubuntu Server 24.04 LTS*, type **t3.medium** (2 vCPU / 4 GB; OCR + vector index need
   the memory), 20 GB gp3 disk, and a key pair.
2. **Security group** inbound rules:
   - SSH (22) from *My IP*;
   - Custom TCP **8000** from *My IP* (or 0.0.0.0/0 for a public demo — change `DEMO_PASSWORD` first).

## 2. Install and start

```bash
ssh -i <key.pem> ubuntu@<EC2-PUBLIC-IP>
git clone https://github.com/surisuraj9/ClinicalChat.git && cd ClinicalChat
bash deployment/ec2-setup.sh
```

The first run creates `.env` from `.env.example` and opens it in an editor. Set:

| Variable | Needed for |
|----------|------------|
| `OPENAI_API_KEY` | the AI agents (Insight, Patient Communication, course specialists) and OpenAI embeddings |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | tracing (optional) |
| `TAVILY_API_KEY` | trusted-site web evidence (optional) |
| `DEMO_PASSWORD` | the persona logins — **change it on a public server** |

Without a key the app still runs with every rule-based feature (lab flags, safety checks, risk, care-team workflow).

## 3. Verify

```bash
curl http://<EC2-PUBLIC-IP>:8000/api/health
curl -u dr.sharma:<DEMO_PASSWORD> http://<EC2-PUBLIC-IP>:8000/api/status
curl -u dr.sharma:<DEMO_PASSWORD> -X POST http://<EC2-PUBLIC-IP>:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Interpret this patient'"'"'s CBC results.", "patient": "ananya-rao-54"}'
curl -u ananya.rao:<DEMO_PASSWORD> http://<EC2-PUBLIC-IP>:8000/api/allergy/peanut
```

Open `http://<EC2-PUBLIC-IP>:8000` for the chat UI and `http://<EC2-PUBLIC-IP>:8000/docs` for the API docs.

## 4. Operate

```bash
sudo systemctl status clinicalchat           # running?
sudo journalctl -u clinicalchat -f           # live logs
sudo systemctl restart clinicalchat          # after editing .env
git pull && bash deployment/ec2-setup.sh     # upgrade (keeps .env and data/)
```

Stopping the service lets the app finish background care-team insights and flush the last Langfuse traces
(`TimeoutStopSec=90`). Patient data (SQLite, uploads, audit log) lives in `data/` — back it up or put it on an EBS
snapshot schedule.

## Docker alternative

```bash
sudo apt install -y docker.io && sudo usermod -aG docker ubuntu   # log out and back in
docker build -t clinicalchat .
docker run -d --name clinicalchat --restart unless-stopped -p 8000:8000 --env-file .env -e PORT=8000 \
           -v clinicalchat-data:/app/data clinicalchat
```
