
import os 

TARGET_DIR = os.path.join(os.getcwd(), "artifacts")

MODEL_KEY = "https://origin-static-assets.s3.us-east-1.amazonaws.com/download/Toto-Open-Base-1.0.zip"
LOCAL_MODEL_DIR = os.path.join(TARGET_DIR, "Toto-Open-Base-1.0")

DATASET_KEY = "https://origin-static-assets.s3.us-east-1.amazonaws.com/download/BOOM.zip"
LOCAL_DATASET_DIR = os.path.join(TARGET_DIR, "BOOM")

