# Render Deployment - Quick Reference

## Service Configuration
- Name: satellite-segmentation
- Environment: Python 3
- Build Command: chmod +x build.sh && ./build.sh
- Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
- Instance Type: Starter ($7/month) minimum

## Environment Variables (Render → Environment)
- GROQ_API_KEY = <paste-your-groq-api-key-from-console.groq.com>
- SECRET_KEY = <use secrets.token_hex(32)>
- FLASK_ENV = production

## Pre-Deployment
- .env in .gitignore
- Model file in repo (<= 100 MB or use Git LFS)

## One-line push
git add . && git commit -m "Deploy to Render" && git push origin main

## Troubleshooting
- OOM → upgrade plan
- Timeout → increase --timeout
- Missing model → ensure .h5 is committed
