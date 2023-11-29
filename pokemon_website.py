from flask import Flask, render_template, request, send_from_directory,jsonify
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix

app = Flask(__name__, static_url_path='/static')



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



model_data = df.copy().drop(columns=['id','name', 'evolves_from', 'generation','type1', 'type2', 'abilities', 'desc'])
model_dataframe = pd.DataFrame(model_data)
    ## replace all mythical pokemon rank with lengendary rank
model_dataframe = model_dataframe.replace("mythical", "legendary")
model_dataframe = model_dataframe.replace("baby", "ordinary")


#====================================================================
#creating test data
big_pokemon_data = []
max_stat_value = 63

for i in range(len(model_dataframe)):
    # Create a copy of the original datapoint
    if model_dataframe.loc[i]['rank'] == 'legendary':
        k = 100
    else:
        k = 10
    for z in range(k):
        new_pokemon = model_dataframe.loc[i].copy()  # Use the copy method to avoid SettingWithCopyWarning

        # Get the randomly chosen datapoint
        tot = 127
        maxx = 63
        count = 0

        stats = [0, 0, 0, 0, 0, 0]
        while count != 0:
            while True:
                random_index = random.randint(2, 7)
                if stats[random_index-2] != max_stat_value:
                    break
            new_stat_value = random.randint(1, max_stat_value - stats[random_index-2] - count)
            stats[random_index-2] += new_stat_value
            count -= new_stat_value

        count = 0
        while count != 0:
            while True:
                random_index = random.randint(2, 7)
                if stats[random_index-2] != max_stat_value:
                    break
            new_stat_value = random.randint(1, max_stat_value - stats[random_index-2] - count)
            stats[random_index-2] += new_stat_value
            count -= new_stat_value

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
    return render_template('index.html', pokemon_list=pokemon_list, pokemon_dict=pokemon_dict)

@app.route('/predict', methods=['POST'])
def predict():
    selected_pokemon = request.form['pokemon']
    selected_pokemon_row = df[df['name'] == selected_pokemon]
    print("rowwww: ",selected_pokemon_row)
    # idd = selected_pokemon_row['id'][1]
    # print("ughhhhhhh:" , selected_pokemon_row['id'][1])
    # print("type: ", type(idd))

    selected_pokemon_row = selected_pokemon_row.drop(columns=['id','name', 'rank','evolves_from', 'generation','type1', 'type2', 'abilities', 'desc'])

    # Scale the features

    # Make predictions using the model
    scaled_features = scaler.transform(selected_pokemon_row)

    prediction = svm_cls.predict(scaled_features)


    # Update the result HTML template with the prediction
    return render_template('result.html', pokemon=selected_pokemon, prediction=prediction)


# Add a route to serve static images
@app.route('/static/<path:image_path>')
def serve_image(image_path):
    return send_from_directory('static', image_path)

if __name__ == '__main__':
    app.run(debug=True)

