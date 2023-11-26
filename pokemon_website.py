from flask import Flask, render_template, request, send_from_directory,jsonify
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from joblib import load

app = Flask(__name__, static_url_path='/static')



svm_cls = load('Pokemon_Predictor.pkl')
scaler = MinMaxScaler(feature_range=(0, 1))



try:
    df = pd.read_csv('pokemons.csv')
    pokemon_list = list(df['name'])

    image_folder = 'static/'
    pokemon_data = df.copy().drop(columns=['id','rank','evolves_from', 'generation','type1', 'type2', 'abilities', 'desc'])


    pokemon_dict = dict(zip(df['name'], [{'total': total, 'image_path': f"/{name.lower()}.png"} for name, total in zip(df['name'], df['total'])]))
except FileNotFoundError:
    print("Error: The 'pokemons.csv' file was not found.")
except pd.errors.EmptyDataError:
    print("Error: The 'pokemons.csv' file is empty.")



@app.route('/')
def home():
    return render_template('index.html', pokemon_list=pokemon_list, pokemon_dict=pokemon_dict)

@app.route('/predict', methods=['POST'])
def predict():
    selected_pokemon = request.form['pokemon']
    selected_pokemon_row = pokemon_data.loc[pokemon_data['name'] == selected_pokemon]

    print("selected poke row", selected_pokemon_row)
    # Scale the features

    selected_pokemon_features = selected_pokemon_row.drop(columns=['name'])

    # Make predictions using the model
    prediction = svm_cls.predict(selected_pokemon_features)

    # Update the result HTML template with the prediction
    return render_template('result.html', pokemon=selected_pokemon, prediction=prediction[0])


# Add a route to serve static images
@app.route('/static/<path:image_path>')
def serve_image(image_path):
    return send_from_directory('static', image_path)

if __name__ == '__main__':
    app.run(debug=True)

