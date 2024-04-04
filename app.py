from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import requests
import csv
import os   

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model import recommend_destination, dataset, travel_features

# MongoDB
from flask_pymongo import PyMongo

app = Flask(__name__)

# MongoDB
app.config["MONGO_URI"] = "mongodb://localhost:27017/newsLetter"  # Replace with your MongoDB URI
mongo = PyMongo(app)

# MongoDB
@app.route('/newsletter_signup', methods=['POST'])
def newsletter_signup():
    email = request.form['email']
    # Insert the email into MongoDB collection
    mongo.db.newsletter_emails.insert_one({'email': email})
    return 'Thank you for signing up for our newsletter!'


# Function to obtain access token
def get_access_token():
    token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    client_id = "Awz0GkL8T3vHjbaXLXJBGpQIHXTQPzyR"
    client_secret = "A6Zekhd2qMan5JGy"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(token_url, data=payload)
    token_data = response.json()
    return token_data.get('access_token')

# For Nearest Destinations Data [START]
@app.route('/get_data', methods=['POST'])
def get_weather():

    city = request.form['pCity']
    radius = request.form['pRadius']
    categories = request.form['pCategory']
    image_path = "img/CityLimits/Limits"+city+".jpg"
    # image_path = "url_for('static', filename='img/Limits" + city + ".jpg"

    # {{ url_for('static', filename='img/j.jpg') }}

    # SIGHTS, NIGHTLIFE, RESTAURANT, SHOPPING

    if city == 'Bangalore':
        latitude = 13.023577
        longitude = 77.536856
    elif city == 'Barcelona':
        latitude=41.42
        longitude=2.11
    elif city == 'Berlin':
        latitude=52.541755
        longitude=13.354201
    elif city == 'Dallas':
        latitude=32.806993
        longitude=-96.836857
    elif city == 'London':
        latitude=51.520180
        longitude=-0.169882
    elif city == 'New York':
        latitude=40.792027
        longitude=-74.058204
    elif city == 'Paris':
        latitude=48.91
        longitude=2.25
    elif city == 'San Francisco':
        latitude = 37.810980
        longitude=-122.483716

    access_token = get_access_token()
    if not access_token:
        return "Failed to obtain access token"

    url = f"https://test.api.amadeus.com/v1/reference-data/locations/pois?latitude={latitude}&longitude={longitude}&radius={radius}&categories={categories}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)

    place_data = response.json()

    # place_name = place_data['data'][0]['name']
    # place_category = place_data['data'][0]['category']
    # place_features = place_data['data'][0]['tags']

    places = []
    for place in place_data['data']:
        place_info = {
          "name": place['name'],
          "category": place['category'],
          "features": ", ".join(place['tags'])
        }
        places.append(place_info)

    return render_template('nearestOutput.html', places=places, image_path=image_path)

# For Nearest Destinations Data [ENDS]


# For Similar Destinations Data [START]
@app.route('/', methods=['GET', 'POST'])
def travel_recommendation():
    if request.method == 'POST':
        user_destination = request.form['destination']
        recommended_destinations = recommend_destination(user_destination, dataset, travel_features)
        # Extract details for recommended destinations
        recommended_details = [list(travel[2:]) for travel in recommended_destinations[1:6]]  # Skip the original destination

        # Get details of user_destination
        user_destination_details = dataset.loc[dataset['Name'] == user_destination].squeeze()

        return render_template('base.html', user_destination=user_destination, user_destination_details=user_destination_details, recommended_details=recommended_details)
    else:
        names = dataset['Name'].tolist()
        return render_template('recommender.html', names=names)
# For Similar Destinations Data [ENDS]
    

if __name__ == '__main__':
    app.run(debug=True)
