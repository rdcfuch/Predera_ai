import json
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load JSON data from file
data = load_json('sorted_city.json')

# Extract keys (text) from the JSON
texts = list(data.keys())

# Initialize lists to store nouns and verbs
nouns = []
verbs = []
adjs=[]

# Process each text
for text in texts:
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "ADJ":
            adjs.append(token.text)

# Print the results

# Filter out single-character items
nouns = [noun for noun in nouns if len(noun) > 1]
verbs = [verb for verb in verbs if len(verb) > 1]
adjs = [adj for adj in adjs if len(adj) > 1]

print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adj:", adjs)
