#!/usr/bin/env bash
# GCE startup script for the gridcast VM (Debian 12). Passed at instance creation with
# --metadata-from-file startup-script=deploy/bootstrap.sh; runs as root at every boot
# and is idempotent (the second run finds everything in place and exits).
#
# What it sets up: Docker Engine + the compose plugin with log rotation, 2 GB of swap
# (e2-small has 2 GB of RAM), unattended security updates, and /opt/gridcast.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${codename} stable" \
    > /etc/apt/sources.list.d/docker.list
  # container logs: 5 x 20 MB per container, so a chatty month cannot fill the 20 GB disk
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<'EOF'
{"log-driver": "json-file", "log-opts": {"max-size": "20m", "max-file": "5"}}
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

if ! swapon --show --noheadings | grep -q '^/swapfile'; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl -w vm.swappiness=10
  grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
fi

if ! dpkg -s unattended-upgrades >/dev/null 2>&1; then
  apt-get install -y unattended-upgrades
fi

mkdir -p /opt/gridcast
echo "bootstrap done"
