#!/bin/bash
# Create HW2-only identities and secret resources; never modifies HW1 services.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/cloud.env
if [[ "${1:-}" != "--run" ]]; then
  echo "Preview: enable APIs; create HW2 build/runtime accounts and four empty secrets."
  echo "Project=$PROJECT_ID Region=$REGION Service=$SERVICE"
  echo "Run bash deploy/setup.sh --run to apply. Requires project IAM/admin permissions."
  exit 0
fi
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project "$PROJECT_ID"
for account in "$RUNTIME_ACCOUNT" "$BUILD_ACCOUNT"; do
  if ! gcloud iam service-accounts describe "$account@$PROJECT_ID.iam.gserviceaccount.com" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$account" --project "$PROJECT_ID" --display-name "$account"
  fi
done
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$BUILD_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com" --role roles/run.builder --condition=None >/dev/null
for secret in hw2-openai-api-key hw2-anthropic-api-key hw2-groq-api-key hw2-tavily-api-key; do
  if ! gcloud secrets describe "$secret" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud secrets create "$secret" --project "$PROJECT_ID" --replication-policy automatic
  fi
  gcloud secrets add-iam-policy-binding "$secret" --project "$PROJECT_ID" --member "serviceAccount:$RUNTIME_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com" --role roles/secretmanager.secretAccessor >/dev/null
done
echo "Setup complete. Add version 1 to each secret in Secret Manager, then deploy."
