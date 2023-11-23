from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__)
df = pd.read_csv('pokemons.csv')
pokemon_dict = dict(zip(df['name'], df['total']))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected Pokemon and its attributes
    selected_pokemon = request.form['pokemon']
    pokemon_attributes_values = pokemon_dict.get(selected_pokemon, {})

    # Pass the attribute values to the template
    return render_template('result.html', pokemon=selected_pokemon, **pokemon_attributes_values)

if __name__ == '__main__':
    app.run(debug=True)
