#!/usr/bin/env bash
# One-command teardown: deletes everything gcp-up.sh created, so nothing keeps billing.
set -uo pipefail

gcloud compute instances delete devmind-cloud --zone=europe-west2-a --quiet 2>/dev/null || true
gcloud compute instances delete devmind-edge-near --zone=europe-west1-b --quiet 2>/dev/null || true
gcloud compute instances delete devmind-edge-far --zone=australia-southeast1-a --quiet 2>/dev/null || true

gcloud compute firewall-rules delete devmind-cloud-access --quiet 2>/dev/null || true
gcloud compute firewall-rules delete devmind-dashboard-access --quiet 2>/dev/null || true

echo "== Remaining compute instances (should be empty) =="
gcloud compute instances list
