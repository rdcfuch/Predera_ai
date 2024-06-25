import json

# Function to convert time duration from 'H:M:S' to seconds
def convert_to_seconds(duration):
    try:
        parts = duration.split(':')
        parts = [int(part) for part in parts]
        if len(parts) == 3:  # Format: H:M:S
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # Format: M:S
            return parts[0] * 60 + parts[1]
        else:  # Invalid format
            return 0
    except ValueError:
        # Handle cases where conversion to int fails
        return 0

# Function to convert seconds to 'H:M:S' format
def convert_to_hms(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02}:{s:02}"

# Load the JSON file
with open('FC_YT_video_list_sea_life_4164_hours.json', 'r', encoding='utf-8') as file:
    videos = json.load(file)

# Calculate the total duration in seconds
total_duration_seconds = sum(convert_to_seconds(video['duration']) for video in videos)

# Convert the total duration back to 'H:M:S' format
total_duration_hms = convert_to_hms(total_duration_seconds)

# Print the total duration
print("Total duration of all videos:", total_duration_hms)
