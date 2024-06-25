import json
from bs4 import BeautifulSoup

# Load the HTML file
with open('youtube_list_6_sea_life.html', 'r', encoding='utf-8') as file:
    content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

# Find all video elements (assuming they are in specific tags with classes for title, href, and duration)
# video_elements = soup.find_all('div', id="contents")  # Adjust the class as per actual HTML structure
video_elements = soup.find_all('div', id="dismissible")  # Adjust the class as per actual HTML structure

print(f"video elements: {len(video_elements)}")
# Extract video titles, href links, and durations
videos = []
for element in video_elements:
    title_element = element.find('a', id='video-title')
    duration_element = element.find("div", class_="badge-shape-wiz__text")
    anchor_tag = element.find("a", class_="yt-simple-endpoint style-scope yt-formatted-string")
    # Extract the text and href attributes
    if anchor_tag:
        text = anchor_tag.get_text()
        href = " https://www.youtube.com"+anchor_tag['href']
        print(f"{text} : {href}")
    if title_element and duration_element:
        url="https://www.youtube.com"+title_element.get('href').split("&")[0]
        video = {
            'title': title_element.get('title'),
            'href': url,
            'duration': duration_element.text.strip()
        }
        videos.append(video)

# Save the video list to a JSON file
with open('FC_YT_video_list_sea_life_4164_hours.json', 'w', encoding='utf-8') as json_file:
    json.dump(videos, json_file, ensure_ascii=False, indent=4)

print("Video list has been saved to video_list.json")
