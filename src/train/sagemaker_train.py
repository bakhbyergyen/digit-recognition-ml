import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
REGION_NAME = os.environ['AWS_REGION']
ROLE_ARN = os.environ['SAGEMAKER_ROLE']
BUCKET_NAME = os.environ['BUCKET_NAME']
OUTPUT_PATH = os.environ['OUTPUT_PATH']
VERSION = os.environ['MODEL_VERSION']


session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)
sagemaker_session = sagemaker.Session(boto_session=session)

# PyTorch Estimator
estimator = PyTorch(
    entry_point='train_script.py',
    source_dir='src/train',
    role=ROLE_ARN,
    framework_version='1.10',
    py_version='py38',
    instance_count=1,
    instance_type='ml.p2.xlarge',
    volume_size=50,
    max_run=3600,
    hyperparameters={
        'batch_size': 32,
        'epochs': 16,
        'lr': 0.01,
        'version': VERSION,
        'bucket_name': BUCKET_NAME,
        'output_path': OUTPUT_PATH
    },
    sagemaker_session=sagemaker_session,
    dependencies=['src/train/requirements.txt'],
)

# Start the training job
estimator.fit()
# Path: src/inference/train/sagemaker_train.py