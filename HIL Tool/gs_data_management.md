### key commands
gcloud auth login


### copy files from gs

- list the file with regex:

gsutil ls gs://predera-developer-testing/yuanjun/visualization/predera-vae-v1.2-480f17-z16-101k-full/ | grep -E 'ff*' 


- copy the files with regex:
gsutil ls gs://predera-developer-testing/yuanjun/visualization/predera-vae-v1.2-480f17-z16-101k-full/ | grep -E 'ff' | xargs -I {} gsutil cp {} ./
