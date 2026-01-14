# íº€ Production Deployment Guide

## Cloud Deployment Options

### Option 1: AWS ECS
```bash
# 1. Push to ECR
aws ecr create-repository --repository-name churn-api
docker tag churn-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/churn-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/churn-api:latest

# 2. Deploy to ECS
# Use AWS Console or Terraform
