#!/usr/bin/env bash
# Bootstrap a single EC2 instance (Ubuntu 22.04+, t3.xlarge or bigger — BERT-large
# needs headroom) with Docker + Minikube, then build and deploy the two
# devmind pods. Run this ON the EC2 instance.
set -euo pipefail

echo "== Installing Docker =="
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "Log out and back in (or 'newgrp docker') for the group change to apply, then re-run this script."
  exit 0
fi

echo "== Installing kubectl =="
if ! command -v kubectl &>/dev/null; then
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/
fi

echo "== Installing minikube =="
if ! command -v minikube &>/dev/null; then
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  chmod +x minikube-linux-amd64
  sudo mv minikube-linux-amd64 /usr/local/bin/minikube
fi

echo "== Starting minikube =="
minikube start --driver=docker --cpus=4 --memory=8g

echo "== Building images inside minikube's docker daemon =="
eval "$(minikube -p minikube docker-env)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -f "$REPO_ROOT/Dockerfile.cloud" -t devmind-cloud:test "$REPO_ROOT"
docker build -f "$REPO_ROOT/Dockerfile.gateway" -t devmind-gateway:test "$REPO_ROOT"

echo "== Applying manifests =="
kubectl apply -f "$REPO_ROOT/k8s/namespace.yaml"
kubectl apply -f "$REPO_ROOT/k8s/configmap.yaml"
kubectl apply -f "$REPO_ROOT/k8s/cloud.yaml"
kubectl apply -f "$REPO_ROOT/k8s/gateway.yaml"

echo "== Waiting for rollout =="
kubectl -n devmind rollout status deployment/devmind-cloud
kubectl -n devmind rollout status deployment/devmind-gateway

echo
echo "Deployed. To reach the gateway from your laptop:"
echo "  1. On the EC2 instance, run in the background:"
echo "       kubectl port-forward -n devmind svc/devmind-gateway 8000:8000 &"
echo "  2. From your laptop, open an SSH tunnel:"
echo "       ssh -L 8000:localhost:8000 <ec2-user>@<ec2-public-ip>"
echo "  3. curl http://localhost:8000/health"
