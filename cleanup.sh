#!/bin/bash
set -e

REGION="${AWS_REGION:-$(aws configure get region)}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ENDPOINT_NAME="sadtalker-async"
BUCKET_NAME="sadtalker-demo-${ACCOUNT_ID}"
REPO_NAME="sadtalker-inference"

echo "Account: ${ACCOUNT_ID}"
echo "Region:  ${REGION}"
echo ""

echo "=== Deleting SageMaker endpoint ==="
aws sagemaker delete-endpoint --endpoint-name "${ENDPOINT_NAME}" --region "${REGION}" 2>/dev/null || true
echo "Waiting for endpoint deletion..."
sleep 10
# Clean up all endpoint configs (v1 and v2)
for cfg in $(aws sagemaker list-endpoint-configs --region "${REGION}" --query "EndpointConfigs[?starts_with(EndpointConfigName,'${ENDPOINT_NAME}')].EndpointConfigName" --output text 2>/dev/null); do
  aws sagemaker delete-endpoint-config --endpoint-config-name "$cfg" --region "${REGION}" 2>/dev/null || true
done
# Clean up all models
for model in $(aws sagemaker list-models --region "${REGION}" --query "Models[?starts_with(ModelName,'${ENDPOINT_NAME}')].ModelName" --output text 2>/dev/null); do
  aws sagemaker delete-model --model-name "$model" --region "${REGION}" 2>/dev/null || true
done

echo "=== Deleting CodeBuild project ==="
aws codebuild delete-project --name sadtalker-build --region "${REGION}" 2>/dev/null || true

echo "=== Deleting ECR repository ==="
aws ecr delete-repository --repository-name "${REPO_NAME}" --force --region "${REGION}" 2>/dev/null || true

echo "=== Deleting S3 bucket ==="
aws s3 rb "s3://${BUCKET_NAME}" --force 2>/dev/null || true

echo "=== Deleting IAM roles ==="
for policy in AmazonEC2ContainerRegistryPowerUser CloudWatchLogsFullAccess AmazonS3FullAccess; do
  aws iam detach-role-policy --role-name CodeBuildSadTalkerRole --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
done
aws iam delete-role --role-name CodeBuildSadTalkerRole 2>/dev/null || true

for policy in AmazonSageMakerFullAccess AmazonS3FullAccess; do
  aws iam detach-role-policy --role-name SageMakerExecutionRole --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
done
aws iam delete-role --role-name SageMakerExecutionRole 2>/dev/null || true

echo ""
echo "✅ Cleanup complete — all resources removed."
