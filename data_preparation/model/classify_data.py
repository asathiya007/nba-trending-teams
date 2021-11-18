import pandas as pd
import csv
from models import predict_random_forest, predict_naive_bayes, predict_linear_regression

# load tweets into dataframe
df = pd.read_csv("../NBAData.csv")

rf_predictions_pos = []
rf_predictions_neg = []
nb_predictions_pos = []
nb_predictions_neg = []
lr_predictions = []

# find predictions from each model
for index, d in df.iterrows():
    if index % 100 == 0:
        print(index)
    rf_prediction = predict_random_forest(d['Tweet'])
    nb_prediction = predict_naive_bayes(d['Tweet'])
    lr_prediction = predict_linear_regression(d['Tweet'])
    
    rf_predictions_pos.append(rf_prediction[0])
    rf_predictions_neg.append(rf_prediction[1])
    nb_predictions_pos.append(nb_prediction[0])
    nb_predictions_neg.append(nb_prediction[1])
    lr_predictions.append(lr_prediction)

# append prediction probabilities into dataframe
df["RF Positive"] = rf_predictions_pos
df["RF Negative"] = rf_predictions_neg
df["NB Positive"] = nb_predictions_pos
df["NB Negative"] = nb_predictions_neg
df["LR Positive"] = lr_predictions

# write to csv
df.to_csv("classifiedNBAData.csv")