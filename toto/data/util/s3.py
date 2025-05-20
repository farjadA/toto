import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any, Hashable

import boto3
import dill
from botocore.config import Config
import json
import subprocess
import time


class S3Client:
    _resource = boto3.resource(
        "s3",
        config=Config(
            connect_timeout=60,
            read_timeout=60,
            retries={"max_attempts": 5},
            s3={"payload_signing_enabled": True, "addressing_style": "auto"},
        ),
    )

    _is_loading_aws_vault_credentials = False

    @classmethod
    def load_aws_vault_credentials(cls, profile: str, max_wait_time: int = 60 * 5) -> None:
        """Load AWS credentials from AWS Vault."""
        # Wait for the previous credential loading to complete
        start_time = time.time()
        while cls._is_loading_aws_vault_credentials:
            if time.time() - start_time > max_wait_time:
                print(
                    f"S3Client load_aws_vault_credentials: Warning: Exceeded maximum wait time ({max_wait_time}s) for credential loading"
                )
                break
            time.sleep(0.1)

        cls._is_loading_aws_vault_credentials = True
        try:
            result = subprocess.run(
                ["aws-vault", "exec", profile, "--json", "--no-session"], capture_output=True, text=True, check=True
            )
            creds = json.loads(result.stdout)

            os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
            os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]
            os.environ["AWS_PROFILE"] = profile
        except Exception as e:
            print(f"Error loading AWS credentials: {e}")
        finally:
            cls._is_loading_aws_vault_credentials = False

    @classmethod
    def reinit_resource(cls) -> None:
        """Refresh the S3 resource."""
        boto3.setup_default_session()
        cls._resource = boto3.resource(
            "s3",
            config=Config(
                connect_timeout=60,
                read_timeout=60,
                retries={"max_attempts": 5},
                s3={"payload_signing_enabled": True, "addressing_style": "auto"},
            ),
        )

    def iter_keys(self, bucket_name: str, prefix: str = "") -> Iterator[str]:
        response = self._resource.meta.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for c in response.get("Contents", ()):
            yield c["Key"]
        while "NextContinuationToken" in response:
            response = self._resource.meta.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=response["NextContinuationToken"],
            )
            for c in response.get("Contents", ()):
                yield c["Key"]

    def list_objects(self, bucket_name: str, prefix: str = "") -> list[str]:
        return list(self.iter_keys(bucket_name, prefix))

    def get(self, bucket_name: str, key: str, raise_on_404: bool = False) -> bytes | None:
        try:
            return self._resource.Object(bucket_name, key).get()["Body"].read()
        except self._resource.meta.client.exceptions.NoSuchKey as e:
            if raise_on_404:
                raise KeyError(key) from e
            return None

    def download_file(self, bucket_name: str, key: str, local_file_path: str) -> None:
        self._resource.meta.client.download_file(bucket_name, key, local_file_path)

    def exists(self, bucket_name: str, key: str) -> bool:
        try:
            self._resource.Object(bucket_name, key).load()
            return True
        except self._resource.meta.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

def copy_dir_from_s3(bucket_name: str, prefix: str, local_dir: Path | str):
    """
    Utility function to copy a directory (prefix) from S3 to a local directory.

    Parameters
    ----------
    bucket_name : str
        The S3 bucket name.
    prefix : str
        The S3 prefix.
    local_dir : Path | str
        The local directory to copy to.
    """
    s3 = S3Client()
    bucket_name = bucket_name.replace("s3://", "")
    for key in s3.iter_keys(bucket_name, prefix):
        try:
            _, filename = os.path.split(key)
        except ValueError:
            continue
        if not filename:
            continue
        local_file_path = os.path.join(local_dir, os.path.relpath(key, prefix))
        if not os.path.isfile(local_file_path):
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            s3.download_file(bucket_name, key, local_file_path)
