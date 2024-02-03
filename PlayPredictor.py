import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):
    data = pd.read_csv(r"C:\Users\rupali\Downloads\PlayPredictor (1).csv")

    print("Size of Actual dataset",len(data))

    feature_names = ['Whether','Temperature']

    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    le = preprocessing.LabelEncoder()

    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print(temp_encoded)

    features = list(zip(weather_encoded,temp_encoded))

    model = KNeighborsClassifier(n_neighbors = 3)

    model.fit(features,label)

    predicted = model.predict([[0,2]])
    print(predicted)

def main():

    print("______Mravellous Infosystems by Piyush Khairnar_____")

    print("Machine Learning Applications")

    print("Play predictor application using K Nearest Kneighbor algorithm")

    MarvellousPlayPredictor(r"C:\Users\rupali\Downloads\PlayPredictor (1).csv")

if __name__ == "__main__":
    main()