# MLOps with Amazon SageMaker and GitHub Actions

## Architecture Overview

![Architecture Diagram](images/architecture-diagram.png)

## Branching Strategy

- main: Production-ready code and deployments.
- develop: Integration of features before merging into `main`.
- feature/: Development of new features or enhancements.

## Instructions

### Setup

### Inference Model

Lambda inference development is not ready. Instead, a simple prediction demonstration is provided in the Jupyter notebook (inference.ipynb) using boto3. This notebook shows how to invoke the SageMaker endpoint for predictions.

### CI/CD Pipeline

The CI/CD pipeline files, `train.yml` and `deploy.yml`, are in the `main` branch. When changes are merged from the `develop` branch, the following steps are triggered:

- **train.yml**: Triggered on pushes to the `main` branch.

  - An AWS PyTorch estimator is executed to create a training job in SageMaker.
  - The model is saved to S3 as defined in `config.yml`.

- **deploy.yml**: Triggered upon successful completion of `train.yml`.
  - Builds a custom Docker image and pushes it to Amazon ECR for inference.
  - A boto3 script is executed to create or update the SageMaker model and endpoint configuration.

## Next Steps

1. Develop Lambda Function: Handle model version control and automate SageMaker endpoint updates. Currently, version control is not fully developed.
2. Distributed Training: The distributed training script is not yet written. The current instance (ml.p2.xlarge) is not designed for distributed training. Multi-GPU instances with spot instances for cost management will be added later.
3. Data Preprocessing: Plan to use Amazon EFS over S3 for faster data retrieval during training. Currently, the setup uses the MNIST dataset.
