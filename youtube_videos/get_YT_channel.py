import json
from bs4 import BeautifulSoup


def check_word_in_document(file_path, word):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Check if the word is in the document content
    return word in content


def deduplicate_channels(channel_list):
    """Deduplicates a list of YouTube channel entries.

    Args:
        channel_list: A list of strings, where each string represents a channel
            entry in the format "Channel Name : https://www.youtube.com/@channel_id"

    Returns:
        A list of unique channel entries.
    """
    # Create a set to store unique channel names (ignoring links)
    unique_channels = set()
    deduplicated_list = []

    for entry in channel_list:
        # Split the entry to get channel name (everything before ":")
        channel_name = entry.split(":")[0].strip()

        # Check if the channel name is already seen
        if channel_name not in unique_channels:
            unique_channels.add(channel_name)
            deduplicated_list.append(entry)

    return deduplicated_list


# Load the HTML file
with open('youtubetest_list_1_demo.html', 'r', encoding='utf-8') as file:
    content = file.read()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')
document_path="current_channel_list.txt"
# Find all video elements (assuming they are in specific tags with classes for title, href, and duration)
# video_elements = soup.find_all('div', id="contents")  # Adjust the class as per actual HTML structure
video_elements = soup.find_all('div', id="dismissible")  # Adjust the class as per actual HTML structure

print(f"video elements: {len(video_elements)}")
# Extract video titles, href links, and durations
channel_list = []
for element in video_elements:
    title_element = element.find('a', id='video-title')
    duration_element = element.find("div", class_="badge-shape-wiz__text")
    anchor_tag = element.find("a", class_="yt-simple-endpoint style-scope yt-formatted-string")
    # Extract the text and href attributes
    if anchor_tag:
        channel_name = anchor_tag.get_text()
        href = anchor_tag.get('href')
        channel_url = " https://www.youtube.com"+anchor_tag['href']


        # Check if the word is in the document
        is_word_present = check_word_in_document(document_path, href)
        # Print the result
        if is_word_present:
            pass
            # print(f"The word '{channel_url}' is present in the document.")
        else:
            # print(f"{channel_name} : {channel_url}")
            channel_list.append(f"{channel_name} : {channel_url}")

deduplicated_channels = deduplicate_channels(channel_list)
for channel in deduplicated_channels:
  print(channel)

