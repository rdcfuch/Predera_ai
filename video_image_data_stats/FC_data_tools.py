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


def check_video_count(input_list, outputfile):
    with open(outputfile, 'a') as file:

        for SEARCH_TAG in input_list:
            sql_string_find_clip_using_tag = f'SELECT clip_id, storage_key, tags FROM clips, UNNEST(tags) AS tag LEFT JOIN date_partitions dt ON clips.date_partitions_id = dt.id WHERE lineage_hash = "28bf24e28cf34a7c81ace1e81f90ada7b50eadb7" AND tag like "%{SEARCH_TAG}%"'

            with db.snapshot() as snapshot:
                results = snapshot.execute_sql(sql_string_find_clip_using_tag)

            # Convert results to list of tuples
            rows = list(results)
            if len(rows) == 0:
                print('No videos found')
                file.write(f'{SEARCH_TAG} video is 0.\n')
            else:
                print(f'Found {SEARCH_TAG}: {len(rows)} videos')
                file.write(f'{SEARCH_TAG} video is {len(rows)}.\n')


Nouns = ['drive', 'board', 'tree', 'money', 'daughter', 'rule', 'person', 'capital', 'class', 'support', 'ground',
         'page', 'queen', 'fish', 'character', 'case', 'city', 'line', 'age', 'road', 'letter', 'figure', 'way',
         'minister', 'living', 'report', 'lake', 'office', 'man', 'bed', 'dad', 'bank', 'water', 'people', 'course',
         'land', 'chief', 'change', 'baby', 'sea', 'fun', 'family', 'room', 'end', 'information', 'place', 'heart',
         'scale', 'price', 'school', 'game', 'property', 'floor', 'brain', 'mouth', 'size', 'account', 'brother',
         'king', 'men', 'tax', 'area', 'color', 'rock', 'house', 'scene', 'eye', 'level', 'book', 'oil', 'card',
         'market', 'hair', 'value', 'doctor', 'ball', 'tv', 'party', 'table', 'phone', 'boy', 'court', 'college',
         'student', 'paper', 'river', 'drop', 'home', 'dog', 'friends', 'glass', 'step', 'body', 'bill', 'child',
         'view', 'department', 'country', 'door', 'car', 'boys', 'fire', 'mind', 'gold', 'computer', 'head', 'food',
         'love', 'son', 'hall', 'field', 'woman', 'range', 'foot', 'work', 'captain', 'feet', 'blood', 'church']
Verbs = ['break', 'drive', 'eat', 'build', 'movie', 'cut', 'lead', 'act', 'waiting', 'cost', 'start', 'thinking',
         'building', 'walk', 'set', 'room', 'seeing', 'face', 'beach', 'getting', 'have', 'form', 'stop', 'play',
         'going', 'find', 'run', 'hand', 'stay', 'dance', 'point', 'touch', 'turn', 'take', 'drink', 'talking', 'learn',
         'eating', 'making', 'writing', 'having']
Adj = ['little', 'front', 'new', 'other', 'royal', 'small', 'open', 'old']

check_video_count(Nouns,"nouns.txt")
check_video_count(Verbs,"verbs.txt")
check_video_count(Adj,"adj.txt")
