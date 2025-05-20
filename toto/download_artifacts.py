from data.util.s3 import copy_dir_from_s3

from global_settings import BUCKET, MODEL_KEY, DATASET_KEY, LOCAL_MODEL_DIR, LOCAL_DATASET_DIR

# Download the model and dataset from S3
copy_dir_from_s3(BUCKET, MODEL_KEY, LOCAL_MODEL_DIR)
copy_dir_from_s3(BUCKET, DATASET_KEY, LOCAL_DATASET_DIR)


