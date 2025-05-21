import os
import zipfile
import urllib.request
from global_settings import MODEL_KEY, DATASET_KEY, TARGET_DIR


def download_file(url, dest_path):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")

def main():
    for url in [MODEL_KEY, DATASET_KEY]:
        filename = os.path.basename(url)
        local_zip = os.path.join(os.getcwd(), filename)

        download_file(url, local_zip)
        unzip_file(local_zip, TARGET_DIR)

if __name__ == "__main__":
    main()