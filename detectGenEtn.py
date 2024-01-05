# import os
# from deepface import DeepFace
# import json

# def save_progress(data, filename):
#     with open(filename, 'w') as f:
#         json.dump(data, f)

# # Load existing data if available
# try:
#     with open('progress.json', 'r') as f:
#         progress = json.load(f)
#     gender_count = progress.get('gender_count', {'Man': 0, 'Woman': 0})
#     races = progress.get('races', [])
# except FileNotFoundError:
#     gender_count = {'Man': 0, 'Woman': 0}
#     races = []

# directory = "/Users/otabekerkinov/Desktop/images/face_age"

# for folder in os.listdir(directory):
#     for file in os.listdir(os.path.join(directory, folder)):
#         img_path = os.path.join(directory, folder, file)

#         try:
#             results = DeepFace.analyze(img_path=img_path, actions=['gender', 'race'])

#             if isinstance(results, list):
#                 for result in results:
#                     gender_count[result['dominant_gender']] += 1
#                     races.append(result['dominant_race'])
#             else:
#                 gender_count[results['dominant_gender']] += 1
#                 races.append(results['dominant_race'])

#             print(f"Current Gender Count: {gender_count}")

#         except Exception as e:
#             print(f"Error processing image {img_path}: {e}")
#             continue

#         # Save progress
#         save_progress({'gender_count': gender_count, 'races': races}, 'progress.json')


import json
import matplotlib.pyplot as plt

# Load the data from progress.json
with open('progress.json', 'r') as file:
    data = json.load(file)

gender_count = data['gender_count']
races = data['races']

# Plotting

# Gender Histogram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(gender_count.keys(), gender_count.values(), color='skyblue')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

# Ethnicity Histogram
plt.subplot(1, 2, 2)
race_count = {race: races.count(race) for race in set(races)}
plt.bar(race_count.keys(), race_count.values(), color='lightgreen')
plt.title('Ethnicity Distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
