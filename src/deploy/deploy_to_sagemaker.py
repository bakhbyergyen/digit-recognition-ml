import boto3
import time
import os


# Initialize session
sagemaker_session = boto3.Session()
client = sagemaker_session.client('sagemaker', region_name=os.getenv('AWS_REGION'))

# variables
ecr_image = os.getenv('ECR_IMAGE')
model_path = os.getenv('MODEL_ARTIFACT')
role = os.getenv('ROLE')
version = os.getenv('MODEL_VERSION')

print(f"Deploying model version: {version}")
print(f"Using image: {ecr_image}")
print(f"Using model artifact: {model_path}")

model_name = f'digital-recognition-model-{version}'
endpoint_name = 'digital-recognition-ml-endpoint'
endpoint_config_name = f'digital-recognition-ml-endpoint-config-{version}'

# Create or update the model
print(f"Creating or updating model with image: {ecr_image} and artifact: {model_path}")
try:
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": ecr_image,
            "ModelDataUrl": model_path
        },
    )
    print(f"Model {model_name} created successfully.")
except client.exceptions.ResourceInUse:
    print(f"Model {model_name} already exists. Updating the existing model.")
    client.update_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image,
            "ModelDataUrl": model_path
        },
    )

# Create or update the endpoint configuration
print(f"Creating or updating endpoint configuration: {endpoint_config_name}")
try:
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )
    print(f"Endpoint configuration {endpoint_config_name} created successfully.")
except client.exceptions.ResourceInUse:
    print(f"Endpoint configuration {endpoint_config_name} already exists. Updating the existing configuration.")
    client.update_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

# Create or update the endpoint
print(f"Creating or updating endpoint: {endpoint_name}")
try:
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print(f"Endpoint {endpoint_name} created successfully.")
except client.exceptions.ResourceInUse:
    print(f"Endpoint {endpoint_name} already exists. Updating the endpoint with the new configuration.")
    client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

def wait_for_endpoint_in_service(endpoint_name):
    print("Waiting for endpoint to be in service")
    while True:
        details = client.describe_endpoint(EndpointName=endpoint_name)
        status = details["EndpointStatus"]
        if status in ["InService", "Failed"]:
            print("\nDone!")
            break
        print(".", end="", flush=True)
        time.sleep(30)

wait_for_endpoint_in_service(endpoint_name)
details = client.describe_endpoint(EndpointName=endpoint_name)
print(details)