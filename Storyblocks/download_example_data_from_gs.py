from google.cloud import storage
import os
def list_and_download_files(bucket_name, prefix, destination_folder):
  """Lists all files in the given bucket with the given prefix and downloads them to the specified destination folder.

  Args:
    bucket_name: The name of the Google Cloud Storage bucket.
    prefix: The prefix to filter the files by.
    destination_folder: The local folder to download the files to.
  """

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)

  blobs = bucket.list_blobs(prefix=prefix)

  for blob in blobs:
    destination_path = os.path.join(destination_folder, blob.name)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    blob.download_to_filename(destination_path)
    print(f"Downloaded {blob.name} to {destination_path}")

# Replace with your bucket name, prefix, and destination folder
bucket_name = "predera-ingest-data"
prefix = "storyblocks/video/"
destination_folder = "/Users/fcfu/Downloads/storyblock_video_samples"

list_and_download_files(bucket_name, prefix, destination_folder)
