# NBA Trending Teams (Data and Machine Learning)

## Overview
This directory contains the data collection, data processing, and machine learning components of the project. This directory contains code for collecting data, processing the data, training machine learning models (logistic regression classifier, multinomial naive Bayes classifier, and random forest classifier), and using those models to label the data for the visualizations. 

## How To Run Locally
1. If you do not have Python3 installed on your machine, please install it from here: `https://www.python.org/downloads/`. 
2. Create a Python3 virtual environment (which will use Python 3.8.5) and install the dependencies in the `../requirements.txt` file using `pip` with the command: `pip install -r ../requirements.txt`. 
3. See the Data Collection, Data Processing and Machine Learning, and Data Labeling sections for using the resources in this directory. 

### Data Collection 
The collected NBA data can be seen in the `NBAData.csv` file. To collect more NBA data, enter the `./model/` directory and run the `gather_data.py` with the command `python3 gather_data.py`. 

### Data Processing and Machine Learning 
To use the default data vectorizers and machine learning models, enter the `./model` directory and unzip the `count_vectorizer.pkl.zip`, `tfidf_transformer.pkl.zip`, `naive_bayes.joblib.zip`, `logistic_regression.joblib.zip`, and `random_forest.joblib.zip` files. 

To retrain the machine learning models, first unzip the `sentiment140dataset.csv.zip` file. Then, run the `models.py` file with the command `python3 models.py`, which will train and save the data vectorizers and machine learning models. Retraining the models takes about 12 minutes. 

A series of experiments have been conducted on the data processing and machine learning components of this project. The code for these experiments is in the `./model/experiments.py` file, and a sample of the experiment results are in the `./model/experiment_results/sample/` directory. The current experiments measure the efficacy and speed of data processing and model fitting, the time for loading saved resources, and the performance of machine learning models on NBA tweets. To rerun the experiments, enter the `./model/` directory and run the `experiments.py` file with the command `python3 experiments.py`. 

### Data Labeling 
To relabel all collected NBA tweets, enter the `./model/` directory and run the `classify_data.py` file with the command `python3 classify_data.py`. Move the output file `classifiedNBAData.csv` outside the model directory so the visualizations can access and use it. 

