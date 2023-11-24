from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os

app = Flask(__name__, static_url_path='/static')

try:
    df = pd.read_csv('pokemons.csv')
    pokemon_list = list(df['name'])
    image_folder = 'static/'
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
    pokemon_attributes_values = pokemon_dict.get(selected_pokemon, {})
    return render_template('result.html', pokemon=selected_pokemon, **pokemon_attributes_values)

# Add a route to serve static images
@app.route('/static/<path:image_path>')
def serve_image(image_path):
    return send_from_directory('static', image_path)

if __name__ == '__main__':
