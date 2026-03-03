#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Deploy Big Island webapp to EC2
# Usage: bash deploy/deploy_bigisland.sh <ec2-host>
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

EC2_HOST="${1:?Usage: deploy_bigisland.sh <ec2-host>}"
REMOTE_DIR="/home/ec2-user/bigisland"

echo "=== Deploying Big Island to $EC2_HOST ==="

# Sync code (exclude local data, .env, caches)
rsync -avz --delete \
    --exclude '.env' \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude '.claude/' \
    --exclude 'docs/' \
    --exclude '*.pyc' \
    ./ "ec2-user@${EC2_HOST}:${REMOTE_DIR}/"

echo "=== Installing deps + restarting services ==="

ssh "ec2-user@${EC2_HOST}" << 'REMOTE'
    cd /home/ec2-user/bigisland

    # Install Python deps
    pip3 install -r requirements-web.txt --break-system-packages 2>/dev/null \
        || pip3 install -r requirements-web.txt

    # Install systemd services
    sudo cp deploy/bigisland.service /etc/systemd/system/
    sudo cp deploy/kona-engine.service /etc/systemd/system/
    sudo cp deploy/nginx_kona.conf /etc/nginx/conf.d/kona.conf
    sudo rm -f /etc/nginx/conf.d/default.conf

    sudo systemctl daemon-reload
    sudo systemctl enable bigisland
    sudo systemctl restart bigisland

    # Reload nginx
    sudo nginx -t && sudo systemctl reload nginx

    echo ""
    echo "=== Checking health ==="
    sleep 3
    curl -sf http://127.0.0.1:8001/healthz && echo " OK" || echo " FAILED"
    echo ""
    echo "Big Island deployed to $HOSTNAME"
REMOTE

echo "=== Deploy complete ==="
