from flask import Flask, render_template, request, send_from_directory,jsonify
import pandas as pd
import os
import random


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix

app = Flask(__name__, static_url_path='/static')



try:
    pokemon_dataframe = pd.read_csv('pokemons.csv') #has all pokemon from csv file with every attribute
    pokemon_names = list(pokemon_dataframe['name']) #list of all pokemon names 
    temp_pokemon_dict = dict(zip(pokemon_dataframe['name'], [{'image_path': f"/{name.lower()}.png"} for name in pokemon_dataframe['name']])) #has a pokemon name and png file name
    pokemon_dict = pd.DataFrame(temp_pokemon_dict) 

except FileNotFoundError:
    print("Error: The 'pokemons.csv' file was not found.")
except pd.errors.EmptyDataError:
    print("Error: The 'pokemons.csv' file is empty.")


#====================================================================
#creating test data

model_data = pokemon_dataframe.copy().drop(columns=['id','name', 'evolves_from', 'generation','type1', 'type2', 'abilities', 'desc'])
model_dataframe = pd.DataFrame(model_data)
    ## replace all mythical pokemon rank with lengendary rank
model_dataframe = model_dataframe.replace("mythical", "legendary")
model_dataframe = model_dataframe.replace("baby", "ordinary")

big_pokemon_data = []

for i in range(len(model_dataframe)):
    # Create a copy of the original datapoint
    if model_dataframe.loc[i]['rank'] == 'legendary':
        k = 100 #artificially putting more legendary pokemon in our dataset so we can classify properly
    else:
        k = 10
    for z in range(k):
        new_pokemon = model_dataframe.loc[i].copy()  # Use the copy method to avoid SettingWithCopyWarning

        # Get the randomly chosen datapoint
        count = 0

        #randomly distributing evs, each pokemon has 127 extra stats which make each species unique.
        #ie pikachu1 will be different than pikachu2 because it has a different stat spread because of evs.
        #so we do this in our code to create our dataset to be as large or as small as we want
        stats = [0, 0, 0, 0, 0, 0]
        for stat_count in range(127):
            while(True):
                rand_stat = random.randint(0,5)
                if stats[rand_stat] < 63: #63 is max evs a pokemon can get per stat
                    stats[rand_stat] += 1
                    break

        for j in range(6):
            new_pokemon[j+2] += stats[j]

        # Append the altered datapoint to the new dataset
        big_pokemon_data.append(new_pokemon)

big_pokemon_data = pd.DataFrame(big_pokemon_data)

y = big_pokemon_data['rank']
set_of_classes = y.value_counts().index.tolist()
set_of_classes= pd.DataFrame({'rank': set_of_classes})
y = pd.get_dummies(y) 
X = big_pokemon_data.drop('rank', axis = 1)


scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X = pd.DataFrame(data = X_rescaled, columns = X.columns)

train, test = train_test_split(big_pokemon_data, test_size=0.2, random_state=21) #splitting the data testsize = 0.2 for 80:20
X_train, y_train = train.drop(columns=['rank']), train['rank'] 
X_test, y_test = test.drop(columns=['rank']), test['rank'] 

print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("Testing set shape (X, y):", X_test.shape, y_test.shape) #showing the 80:20 split



scaler = MinMaxScaler(feature_range=(0, 1))
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled = scaler.transform(X_test)


print(X_train_rescaled)
# Create an SVM classifier
svm_cls = SVC(kernel='linear', C=1.0)
svm_cls.fit(X_train_rescaled, y_train)

# Make predictions on the test set
y_pred = svm_cls.predict(X_test_rescaled)


# Evaluate the SVM model
print("SVM Classification Report:")
report = classification_report(y_test, y_pred)
print(report)


#creating model
#====================================================================

@app.route('/')
def home():
    return render_template('index.html', pokemon_names=pokemon_names, pokemon_dict=pokemon_dict)

@app.route('/predict', methods=['POST'])
def predict():
    selected_pokemon = request.form['pokemon']
    selected_pokemon_row = pokemon_dataframe[pokemon_dataframe['name'] == selected_pokemon]

   
    index = pokemon_dataframe[pokemon_dataframe['name'] == selected_pokemon].index[0]
   # print(f"The index of {selected_pokemon} is: {index}")


    selected_pokemon_row = selected_pokemon_row.drop(columns=['id','name', 'rank','evolves_from', 'generation','type1', 'type2', 'abilities', 'desc'])

    # Scale the features
    #randomly distributing evs, each pokemon has 127 extra stats which make each species unique.
    #ie pikachu1 will be different than pikachu2 because it has a different stat spread because of evs.
    #so we do this in our code to create our dataset to be as large or as small as we want
    stats = [0, 0, 0, 0, 0, 0]
    for stat_count in range(127):
            while(True):
                rand_stat = random.randint(0,5)
                if stats[rand_stat] < 63: #63 is max evs a pokemon can get per stat
                    stats[rand_stat] += 1
                    break
    #randomly distributing evs, for each unique pokemon so it is consistent with our data. 
    selected_pokemon_row['hp'] += stats[0]
    selected_pokemon_row['atk'] += stats[1]
    selected_pokemon_row['def'] += stats[2]
    selected_pokemon_row['spatk'] += stats[3]
    selected_pokemon_row['spdef'] += stats[4]
    selected_pokemon_row['speed'] += stats[5]


    # Make predictions using the model
    scaled_features = scaler.transform(selected_pokemon_row)

    prediction = svm_cls.predict(scaled_features)


    # Update the result HTML template with the prediction
    return render_template('result.html', prediction=prediction,pokemon_names=pokemon_names, pokemon_dict=pokemon_dict,index = index, pokemon=selected_pokemon)
    #pokemon_names has all names, pokemon_dict has names, and image path, index is index needed

# Add a route to serve static images
@app.route('/static/<path:image_path>')
def serve_image(image_path):
    return send_from_directory('static', image_path)

if __name__ == '__main__':
    app.run(debug=True)

