import torch
import tarfile
import os
import boto3

def save_model(model, save_path):
    """
    Save the PyTorch model to the specified path.
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def create_tar_gz(source_path, tar_path):
    """
    Create a tar.gz file containing the specified source file.
    """
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(source_path, arcname=os.path.basename(source_path))
    print(f"Model tar.gz file created at {tar_path}")

def upload_to_s3(file_path, bucket_name, s3_path):
    """
    Upload the specified file to S3.
    """
    s3 = boto3.client(
        "s3"
    )
    s3.upload_file(file_path, bucket_name, s3_path)
    print(f"Model uploaded to s3://{bucket_name}/{s3_path}")

def clean_up(*file_paths):
    """
    Remove the specified files.
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
