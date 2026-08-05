#!/usr/bin/env bash
# Resume a stack paused with gcp-stop.sh. Re-adds an ephemeral external IP
# to each VM (static ones were released to avoid idle-IP charges), starts
# them, then re-points the firewall + edge containers at the new addresses.
# Docker images are cached on each disk, so this is a fast container
# recreate, not a rebuild.
set -euo pipefail

CLOUD_ZONE="europe-west2-a"; NEAR_ZONE="europe-west1-b"; FAR_ZONE="australia-southeast1-a"
CLOUD_NAME="devmind-cloud"; NEAR_NAME="devmind-edge-near"; FAR_NAME="devmind-edge-far"

start_instance() {
  local name="$1" zone="$2"
  local ac_name
  ac_name="$(gcloud compute instances describe "$name" --zone="$zone" --format='get(networkInterfaces[0].accessConfigs[0].name)' 2>/dev/null || true)"
  if [[ -z "$ac_name" ]]; then
    gcloud compute instances add-access-config "$name" --zone="$zone" --access-config-name="external-nat"
  fi
  gcloud compute instances start "$name" --zone="$zone"
}

start_instance "$CLOUD_NAME" "$CLOUD_ZONE"
start_instance "$NEAR_NAME" "$NEAR_ZONE"
start_instance "$FAR_NAME" "$FAR_ZONE"

echo "== Waiting for SSH =="
wait_for_ssh() {
  local name="$1" zone="$2"
  for _ in $(seq 1 30); do
    gcloud compute ssh "$name" --zone="$zone" --command="true" &>/dev/null && return 0
    sleep 10
  done
  echo "Timed out waiting for SSH on $name" >&2
  exit 1
}
wait_for_ssh "$CLOUD_NAME" "$CLOUD_ZONE"
wait_for_ssh "$NEAR_NAME" "$NEAR_ZONE"
wait_for_ssh "$FAR_NAME" "$FAR_ZONE"

CLOUD_IP="$(gcloud compute instances describe "$CLOUD_NAME" --zone="$CLOUD_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
NEAR_IP="$(gcloud compute instances describe "$NEAR_NAME" --zone="$NEAR_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
FAR_IP="$(gcloud compute instances describe "$FAR_NAME" --zone="$FAR_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo "== Re-pointing firewall at the new edge IPs =="
gcloud compute firewall-rules update devmind-cloud-access --source-ranges="${NEAR_IP}/32,${FAR_IP}/32"

MY_IP_RAW="${MY_IP:-$(curl -4 -s --max-time 5 https://ifconfig.me || true)}"
if [[ "$MY_IP_RAW" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
  gcloud compute firewall-rules update devmind-dashboard-access --source-ranges="${MY_IP_RAW}/32"
else
  echo "Could not confirm your public IP -- dashboard firewall rule left unchanged." >&2
fi

echo "== Re-pointing edge containers at the new cloud IP (cached image, no rebuild) =="
gcloud compute ssh "$NEAR_NAME" --zone="$NEAR_ZONE" --command="
  for spec in 'gw-nhs:8000:client_nhs' 'gw-streamforge:8010:client_streamforge' 'gw-newco:8020:client_newco'; do
    IFS=: read -r cname port client <<< \"\$spec\"
    sudo docker rm -f \"\$cname\" 2>/dev/null || true
    sudo docker run -d --restart unless-stopped -p \"\$port:\$port\" \
      -e DEVMIND_CLOUD_URL=http://${CLOUD_IP}:8001 -e DEVMIND_CLIENT_ID=\"\$client\" -e DEVMIND_PORT=\"\$port\" \
      --name \"\$cname\" devmind-gateway
  done
"
gcloud compute ssh "$FAR_NAME" --zone="$FAR_ZONE" --command="
  sudo docker rm -f gw-babcock 2>/dev/null || true
  sudo docker run -d --restart unless-stopped -p 8000:8000 \
    -e DEVMIND_CLOUD_URL=http://${CLOUD_IP}:8001 -e DEVMIND_CLIENT_ID=client_babcock -e DEVMIND_PORT=8000 \
    --name gw-babcock devmind-gateway
"

cat <<SUMMARY

== Resumed. ==
Dashboard: http://${CLOUD_IP}:8002
SUMMARY
