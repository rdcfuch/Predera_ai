import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to save JSON data to a file
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Load JSON data from file
data = load_json('city.json')

# Sort the JSON data by values
sorted_data = dict(sorted(data.items(), key=lambda item: float(item[1]), reverse=True))

# Save the sorted JSON data to a new file
save_json(sorted_data, 'sorted_city.json')

# Print the sorted JSON data
print(json.dumps(sorted_data, indent=4))
