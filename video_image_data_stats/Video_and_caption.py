import json

from google.cloud import storage
from google.colab import auth
from collections import defaultdict
from google.cloud import spanner
import pandas as pd
import io
import base64
from IPython.display import HTML
from matplotlib import pyplot as plt
import numpy as np
import getpass

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

'''
you must do : 

%  gcloud auth application-default login

% export GOOGLE_APPLICATION_CREDENTIALS=/Users/fcfu/.config/gcloud/application_default_credentials.json

'''

# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
auth.authenticate_user()

spanner_client = spanner.Client(project="prod-423420")
instance = spanner_client.instance(instance_id="predera-spanner")
db = instance.database(database_id="data-acquisition-database")





project_id = 'prod-423420'
bucket_name = 'predera-clips'

SEARCH_TAG = 'hippo'
sql_string_find_clip_using_tag = f'SELECT clip_id, storage_key, tags FROM clips, UNNEST(tags) AS tag LEFT JOIN date_partitions dt ON clips.date_partitions_id = dt.id WHERE lineage_hash = "28bf24e28cf34a7c81ace1e81f90ada7b50eadb7" AND tag like "%{SEARCH_TAG}%"'

with db.snapshot() as snapshot:
  results = snapshot.execute_sql(sql_string_find_clip_using_tag)

  # Convert results to list of tuples
  rows = list(results)
if len(rows) == 0:
  print('No videos found')
else:
  print(f'Found {len(rows)} videos')
video_file_names = [r[1]+'/video' for r in rows]
caption_file_names = [r[1]+'/metadata' for r in rows]



storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)
# videos_per_row = 2
num_show_videos = 1 #Cannot show more than 2 videos currently
video_htmls = []
for i, video_file_name in enumerate(video_file_names[4020:4021]):
  blob = bucket.blob(video_file_name)
  video_content = blob.download_as_bytes()
  video_base64 = base64.b64encode(video_content).decode('utf-8')
  caption_blob = bucket.blob(caption_file_names[4021])
  caption_content = caption_blob.download_as_text()

  # Parse the JSON
  data = json.loads(caption_content)
  # Parse the nested JSON in llava_next_caption
  llava_caption = json.loads(data['llava_next_caption'])







  video_html = f'''
    <figure>
      <video width="320" height="240" controls muted>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
      </video>

      <body>
    <h1>Video Metadata</h1>
    <div class="metadata">
        <p><strong>Container Format:</strong> {data['container_format']}</p>
        <p><strong>Duration:</strong> {data['duration']} seconds</p>
        <p><strong>Start Time:</strong> {data['start_time']}</p>
        <p><strong>End Time:</strong> {data['end_time']}</p>
        <h2>Encoded Fields:</h2>
        <ul>
            {"".join(f"<li>{field}</li>" for field in data['encoded_fields'])}
        </ul>
        <h2>Caption:</h2>
        <p>{llava_caption['description']}</p>
        <h3>Tags:</h3>
        <div>
            {"".join(f'<span class="tag">{tag},</span>' for tag in llava_caption['tags'])}
        </div>
        <h2>Additional Information:</h2>
        <p><strong>llava_next_caption_t5_embedding_shape:</strong> {data['llava_next_caption_t5_embedding_shape']}</p>
        <p><strong>llava_next_caption_t5_mask_shape:</strong> {data['llava_next_caption_t5_mask_shape']}</p>
    </div>
</body>
    </figure>
  '''
  # video_html = f'<video width="320" height="240" controls muted><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>'
  video_htmls.append(video_html)

# Display videos in the specified layout
HTML(''.join(video_htmls))

# put into file
html_content = ''.join(video_htmls)

with open('videos.html', 'w') as f:
    f.write(html_content)

