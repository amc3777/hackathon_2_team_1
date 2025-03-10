#!/bin/bash

# Environment Variables (Set these before running the script)
PROJECT_ID="${PROJECT_ID:?PROJECT_ID environment variable not set.}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:?SERVICE_ACCOUNT environment variable not set.}"
REGION="${REGION:?REGION environment variable not set.}"
REPOSITORY_NAME="streamlit-repo"
IMAGE_NAME="streamlit-app"
CONTAINER_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}"

if ! gcloud artifacts repositories describe "$REPOSITORY_NAME" --location="$LOCATION" --project="$PROJECT_ID" > /dev/null 2>&1; then
  echo "Artifact Registry repository '$REPOSITORY_NAME' does not exist. Creating..."
  gcloud artifacts repositories create "$REPOSITORY_NAME" \
    --location="$REGION" \
    --repository-format="DOCKER" \
    --project="$PROJECT_ID" \
    --description="Docker repository for my project"
else
  echo "Artifact Registry repository '$REPOSITORY_NAME' already exists."
fi

# Build the Docker Image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Authenticate Docker to Google Cloud
echo "Authenticating Docker to Google Cloud..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Tag the Docker Image
echo "Tagging Docker image..."
docker tag ${IMAGE_NAME} ${CONTAINER_IMAGE}

# Push the Docker Image to Artifact Registry
echo "Pushing Docker image to Artifact Registry..."
docker push ${CONTAINER_IMAGE}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${IMAGE_NAME} \
    --image ${CONTAINER_IMAGE} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --service-account ${SERVICE_ACCOUNT}

echo "Deployment complete."