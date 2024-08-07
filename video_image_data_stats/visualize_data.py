import json
import matplotlib.pyplot as plt

# Step 1: Read data from JSON file
with open('video_data.json', 'r') as file:
    data = json.load(file)


# Sort data by values
sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))

# Prepare data for plotting
labels = list(sorted_data.keys())
values = list(sorted_data.values())

print(sorted_data)

# Create the plot
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Data Visualization')
plt.xticks(rotation=45)
plt.show()

