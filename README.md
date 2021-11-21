# NBA Trending Teams Map
Project repository for Georgia Tech CS 6242 course project. Machine learning and data visualization are used to show the most/least popular NBA teams in each US state. 

# Getting Started 
Create a python virtual environment called `nbattmap` and install the dependencies in the `requirements.txt` file using `pip`. 

Enter the `./data_preparation/model` directory and unzip the `count_vectorizer.pkl.zip`, `tfidf_transformer.pkl.zip`, `naive_bayes.joblib.zip`, `logistic_regression.joblib.zip`, and `random_forest.joblib.zip` files to use the default vectorizers and machine learning models. To retrain the models, first unzip the `sentiment140dataset.csv.zip` file. Then, run the `models.py` file with the command `python3 models.py`, which will train and save the data vectorizers and machine learning models. Retraining the models takes about 12 minutes. 

# Experiments 
A series of experiments have been conducted on the data processing and machine learning components of this project. The code for these experiments is in the `./data_preparation/model/experiments.py` file, and a sample of the experiment results are in the `./data_preparation/model/experiment_results/sample` directory. 