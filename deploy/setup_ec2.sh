#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Kona v56 - EC2 Bootstrap Script
# Run on a fresh Amazon Linux 2023 or Ubuntu 22.04 instance
# Usage: sudo bash setup_ec2.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="/home/ec2-user/kona_aws"
APP_USER="ec2-user"

echo "=== Kona EC2 Setup ==="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

echo "Detected OS: $OS"

# Install system packages
if [ "$OS" = "amzn" ]; then
    dnf update -y
    dnf install -y python3.11 python3.11-pip nginx git
elif [ "$OS" = "ubuntu" ]; then
    apt-get update -y
    apt-get install -y python3.11 python3.11-venv python3-pip nginx git
else
    echo "Unsupported OS: $OS. Install Python 3.11+, pip, nginx, git manually."
    exit 1
fi

# Create app directory
mkdir -p "$APP_DIR"
chown "$APP_USER:$APP_USER" "$APP_DIR"

# Install Python dependencies
cd "$APP_DIR"
pip3 install --upgrade pip
pip3 install flask gunicorn

# Copy systemd service
cp deploy/kona.service /etc/systemd/system/kona.service
systemctl daemon-reload
systemctl enable kona

# Copy nginx config
cp deploy/nginx_kona.conf /etc/nginx/conf.d/kona.conf
# Remove default nginx config if present
rm -f /etc/nginx/conf.d/default.conf /etc/nginx/sites-enabled/default
nginx -t && systemctl enable nginx && systemctl restart nginx

# Install crontab
crontab -u "$APP_USER" deploy/crontab_kona

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy your .env file to $APP_DIR/.env"
echo "  2. Copy your universe CSV to $APP_DIR/"
echo "  3. Run: sudo systemctl start kona"
echo "  4. Check: sudo systemctl status kona"
echo "  5. View logs: sudo journalctl -u kona -f"
echo ""
