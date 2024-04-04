import numpy as np
import pandas as pd
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset_file = pd.read_csv('europe_travel_destinations.csv')
dataset = []

dataset_file = os.path.join(os.getcwd(), 'dataset.csv')

with open('europe_travel_destinations.csv', 'r', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    dataset.append(row)

# Converting the list to a DataFrame
dataset = pd.DataFrame(dataset)

# Select the desired columns
dataset = dataset.loc[:, ['Name', 'Country', 'Type', 'Climate', 'Activities', 'Latitude', 'Longitude', 'Cost', 'Rating']]

vectorizer = TfidfVectorizer()

travel_country = []
travel_type = []
travel_climate = []
travel_activities = []

for travel in dataset:
  travel_country = dataset["Country"].tolist()
  travel_type = dataset["Type"].tolist()
  travel_climate = dataset["Climate"].tolist()
  travel_activities = dataset["Activities"].tolist()

country_matrix = vectorizer.fit_transform(travel_country).toarray()
type_matrix = vectorizer.fit_transform(travel_type).toarray()
climate_matrix = vectorizer.fit_transform(travel_climate).toarray()
activities_matrix = vectorizer.fit_transform(travel_activities).toarray()

numerical_data = dataset.iloc[:, 5:].values

for matrix in [country_matrix, type_matrix, climate_matrix, activities_matrix]:
    if len(matrix.shape) > 1:
        matrix = matrix.reshape(-1)

travel_features = np.hstack([country_matrix, type_matrix, climate_matrix, activities_matrix, numerical_data])

def recommend_destination(destination_title, travel_dataset, travel_destination_features, similarity_metric=cosine_similarity):

  # Find the index of the destination in the dataset by name
  travel_index = None
  for i in range(len(travel_dataset)):
    travel = travel_dataset.iloc[i]  # Access travel data point directly by index
    if travel["Name"] == destination_title:
      travel_index = i
      break  # Stop iterating once found

  # Ensure a destination was found
  if travel_index is None:
      raise ValueError(f"Destination '{destination_title}' not found in dataset")

  # Calculate similarity scores between the chosen destination and all others
  similarity_scores = similarity_metric(travel_features[travel_index].reshape(1, -1), travel_features)

  # Pair each destination with its similarity score
  destinations_with_scores = list(zip(range(len(travel_dataset)), similarity_scores[0], travel_dataset['Name'], travel_dataset['Country'], travel_dataset['Type'], travel_dataset['Climate'], travel_dataset['Activities'], travel_dataset['Latitude'], travel_dataset['Longitude'], travel_dataset['Cost'], travel_dataset['Rating']))

  # Sort destinations and their corresponding scores together (by score, descending)
  sorted_similar_destinations = sorted(destinations_with_scores, key=lambda x: x[1], reverse=True)

  return sorted_similar_destinations

input_arr = ['Destination2']

travel_details_arr = recommend_destination(input_arr[0], dataset, travel_features)

# Extract names of the 5 most similar destinations (excluding the original one)
recommended_destinations = []
for travel in travel_details_arr[1:6]:
  recommended_destinations.append(travel[2:])

print(recommended_destinations)