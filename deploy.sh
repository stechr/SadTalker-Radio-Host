#!/bin/bash
set -e

# Auto-detect from AWS CLI config
REGION="${AWS_REGION:-$(aws configure get region)}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

REPO_NAME="sadtalker-inference"
BUCKET_NAME="sadtalker-demo-${ACCOUNT_ID}"
ENDPOINT_NAME="sadtalker-async"
INSTANCE_TYPE="ml.g5.xlarge"
IMAGE_TAG="latest"
FULL_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "Account: ${ACCOUNT_ID}"
echo "Region:  ${REGION}"
echo ""

echo "=== Step 1: Create S3 bucket ==="
aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}" 2>/dev/null || echo "Bucket already exists"

echo "=== Step 2: Create ECR repository ==="
aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" 2>/dev/null || echo "Repo already exists"

echo "=== Step 3: Create IAM roles ==="
aws iam create-role --role-name CodeBuildSadTalkerRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' 2>/dev/null || echo "CodeBuild role exists"

aws iam attach-role-policy --role-name CodeBuildSadTalkerRole --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser 2>/dev/null || true
aws iam attach-role-policy --role-name CodeBuildSadTalkerRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess 2>/dev/null || true
aws iam attach-role-policy --role-name CodeBuildSadTalkerRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

aws iam create-role --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' 2>/dev/null || echo "SageMaker role exists"

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess 2>/dev/null || true
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true

echo "Waiting for role propagation..."
sleep 10

echo "=== Step 4: Upload source and build Docker image via CodeBuild ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../container"
zip -r /tmp/sadtalker-build.zip Dockerfile serve.py buildspec.yml
aws s3 cp /tmp/sadtalker-build.zip "s3://${BUCKET_NAME}/build/source.zip"

aws codebuild create-project \
  --name sadtalker-build \
  --source "{\"type\":\"S3\",\"location\":\"${BUCKET_NAME}/build/source.zip\"}" \
  --artifacts '{"type":"NO_ARTIFACTS"}' \
  --environment '{
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/standard:7.0",
    "computeType": "BUILD_GENERAL1_LARGE",
    "privilegedMode": true
  }' \
  --service-role "arn:aws:iam::${ACCOUNT_ID}:role/CodeBuildSadTalkerRole" \
  --region "${REGION}" 2>/dev/null || echo "CodeBuild project exists"

echo "Starting build (this takes ~10 minutes)..."
BUILD_ID=$(aws codebuild start-build --project-name sadtalker-build --region "${REGION}" --query 'build.id' --output text)
echo "Build ID: ${BUILD_ID}"

# Poll until build completes
while true; do
  STATUS=$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${REGION}" --query 'builds[0].buildStatus' --output text)
  if [ "$STATUS" = "SUCCEEDED" ]; then
    echo "✅ Build succeeded!"
    break
  elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "STOPPED" ]; then
    echo "❌ Build ${STATUS}. Check CloudWatch logs."
    exit 1
  fi
  echo "  Build status: ${STATUS}... waiting"
  sleep 30
done

echo "=== Step 5: Deploy SageMaker endpoint ==="
aws sagemaker create-model \
  --model-name "${ENDPOINT_NAME}" \
  --primary-container "Image=${FULL_IMAGE}" \
  --execution-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/SageMakerExecutionRole" \
  --region "${REGION}"

aws sagemaker create-endpoint-config \
  --endpoint-config-name "${ENDPOINT_NAME}" \
  --production-variants "[{
    \"VariantName\": \"default\",
    \"ModelName\": \"${ENDPOINT_NAME}\",
    \"InstanceType\": \"${INSTANCE_TYPE}\",
    \"InitialInstanceCount\": 1
  }]" \
  --async-inference-config "{
    \"OutputConfig\": {
      \"S3OutputPath\": \"s3://${BUCKET_NAME}/async-output/\"
    }
  }" \
  --region "${REGION}"

aws sagemaker create-endpoint \
  --endpoint-name "${ENDPOINT_NAME}" \
  --endpoint-config-name "${ENDPOINT_NAME}" \
  --region "${REGION}"

echo ""
echo "⏳ Endpoint deploying (takes ~10 minutes)..."
echo "   Check: aws sagemaker describe-endpoint --endpoint-name ${ENDPOINT_NAME} --query EndpointStatus"
echo ""

# Poll until endpoint is InService
while true; do
  EP_STATUS=$(aws sagemaker describe-endpoint --endpoint-name "${ENDPOINT_NAME}" --region "${REGION}" --query 'EndpointStatus' --output text)
  if [ "$EP_STATUS" = "InService" ]; then
    echo "✅ Endpoint is InService!"
    break
  elif [ "$EP_STATUS" = "Failed" ]; then
    echo "❌ Endpoint failed."
    exit 1
  fi
  echo "  Endpoint status: ${EP_STATUS}... waiting"
  sleep 30
done

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  Endpoint: ${ENDPOINT_NAME}"
echo "  Bucket:   ${BUCKET_NAME}"
echo "  Region:   ${REGION}"
echo ""
echo "  Next: cd webapp && python app.py"
echo "════════════════════════════════════════════"
