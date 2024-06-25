import json
from bs4 import BeautifulSoup

# Load the HTML file
with open('youtubetest_list_1_demo.html', 'r', encoding='utf-8') as file:
    content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

# Find all video elements (assuming they are in anchor tags with class 'video-title')
videos = soup.find_all('a', id='video-title')
# Find the element containing the time text
# time_elements = soup.find_all("div", class_="badge-shape-wiz__text")
# Extract video names and URLs
video_list = []
for video in videos:
    # print(video['href'])
    video_name = video.get('title')
    video_url = video.get('href')
    video_url = video_url.split('&')[0]
    video_url = 'https://www.youtube.com/' + video_url
    if video_name and video_url:
        video_list.append({'name': video_name, 'url': video_url})

# Save the video list to a JSON file
with open('video_list.json', 'w', encoding='utf-8') as json_file:
    json.dump(video_list, json_file, ensure_ascii=False, indent=4)

print("Video list has been saved to video_list.json")
