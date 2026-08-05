#!/usr/bin/env bash
# Stop the eval stack AND release the static IPs / access configs, so nothing
# keeps billing while paused -- a stopped instance still counts as "not
# running," so a reserved-but-idle static IP bills ~$0.01/hr on its own.
# Docker images/layers on each boot disk are untouched, so gcp-start.sh
# resumes without reinstalling anything -- just a container recreate.
set -uo pipefail

stop_and_release() {
  local name="$1" zone="$2" region="$3"
  gcloud compute instances stop "$name" --zone="$zone"
  local ac_name
  ac_name="$(gcloud compute instances describe "$name" --zone="$zone" --format='get(networkInterfaces[0].accessConfigs[0].name)' 2>/dev/null || true)"
  if [[ -n "$ac_name" ]]; then
    gcloud compute instances delete-access-config "$name" --zone="$zone" --access-config-name="$ac_name" 2>/dev/null || true
  fi
  gcloud compute addresses delete "${name}-ip" --region="$region" --quiet 2>/dev/null || true
}

stop_and_release devmind-cloud europe-west2-a europe-west2
stop_and_release devmind-edge-near europe-west1-b europe-west1
stop_and_release devmind-edge-far australia-southeast1-a australia-southeast1

echo "== Stopped, static IPs released. Only the ~50GB boot disks still bill (storage-rate). =="
gcloud compute instances list
