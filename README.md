# NBA Trending Teams
Project repository for Georgia Tech CS 6242 course project. This project provides a visual representation of positively and negatively trending NBA teams over time based on Twitter sentiment analysis. This visualization will provide NBA executives, marketers, and ad agencies with crucial insights to aid in future advertising, exhibition games, expansion programs, etc.

There are three main components to this project: the data/machine learning component, the line graph visualization, and the bar graph visualization. 

# NBA Trending Teams (Data and Machine Learning)

## Overview
The data and machine learning component of the project contains the code for collecting data, processing the data, training machine learning models (logistic regression classifier, multinomial naive Bayes classifier, and random forest classifier), and using those models to label the data for the visualizations. 

## How To Run Locally
1. If you do not have Python3 installed on your machine, please install it from here: `https://www.python.org/downloads/`. 
2. Create a Python3 virtual environment (which will use Python 3.8.5) using `pip` (as described here: `https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/`) and install the dependencies in the `./requirements.txt` file with the command: `pip install -r ./requirements.txt`. 
3. See the Data Collection, Data Processing and Machine Learning, and Data Labeling sections for using the resources in this directory. 

### Data Collection 
The collected NBA data can be seen in the `NBAData.csv` file. To collect more NBA data, enter the `./data_preparation/model/` directory and run the `gather_data.py` file with the command `python3 gather_data.py`. 

### Data Processing and Machine Learning 
To use the default data vectorizers and machine learning models, enter the `.data_preparation/model` directory and unzip the `count_vectorizer.pkl.zip`, `tfidf_transformer.pkl.zip`, `naive_bayes.joblib.zip`, `logistic_regression.joblib.zip`, and `random_forest.joblib.zip` files. 

To retrain the machine learning models, first unzip the `sentiment140dataset.csv.zip` file. Then, run the `models.py` file with the command `python3 models.py`, which will train and save the data vectorizers and machine learning models. Retraining the models takes about 12 minutes. 

A series of experiments have been conducted on the data processing and machine learning components of this project. The code for these experiments is in the `.data_preparation/model/experiments.py` file, and a sample of the experiment results are in the `.data_preparation/model/experiment_results/sample/` directory. The current experiments measure the efficacy and speed of data processing and model fitting, the time for loading saved resources, and the performance of machine learning models on NBA tweets. To rerun the experiments, enter the `./model/` directory and run the `experiments.py` file with the command `python3 experiments.py`. 

### Data Labeling 
To relabel all collected NBA tweets, enter the `.data_preparation/model/` directory and run the `classify_data.py` file with the command `python3 classify_data.py`. Move the output file `classifiedNBAData.csv` outside the model directory so the visualizations can access and use it. 

# Line Graph Visualization

## Overview
This visualization shows an overall sentiment trend for each of the NBA teams over a specified period of time.

Our trained models were used to classify the sentiment for each of the Tweets as either positive or negative. Based on this, each of the NBA teams has a positive Tweets count and a negative Tweets count for every day that data was collected.

The nature of this data led to two graphs being displayed on the HTML page: a positive Tweet counts vs. date graph and a negative Tweet counts vs. date graph.

## How To Run Locally
1. Clone this repository to your local machine.
2. Navigate to the `line_graph_visualization` directory within the cloned repo through your command-line interface.
3. Run the following command to start up a local server:
    ```bash
    python3 -m http.server
    ```
    > This command will start up a server at http://localhost:8000. 
4. To view the line graph visualization, paste the following link into a web browser of your choosing: http://localhost:8000/line_graphs.html

## Functionality Details
This interface allows for deeper analysis through selecting specific teams and time ranges.

### Menu Options:
- Model Select
  - This option allows you to visualize the sentiment classification results of three different models: random forest, naive bayes, and logistic regression. Selecting between the three models will show that each of the models classifies the tweets with slight differences.
- Team Select
  - This option is where you can select the teams you specifically want to visualize on the graphs. You can select as many teams as you would like. On macOS, use `Shift` to select a consecutive group of teams or `Command` to select nonconsecutive teams.
- Min Date Select
  - Use this option to specify a minimum date for the x-axis.
- Max Date Select
  - Use this option to specify a maximum date for the x-axis.
  
### Tool Tip Information:
Hover your mouse over a circular data point to view the team name and sample Tweet associated with that data point. The sample Tweet is pulled from the positive/negative group of Tweets for that team on that particular date.

For example, hovering your mouse over the Blazers' datapoint on `2021-11-11` within the positive Tweets graph will display a tool tip containing the Blazers' team name along with a positively-classified Blazers Tweet from November 11, 2021. 

# Bar Graph Visualization

## Overview
This visualization displays the sentiments of various NBA teams, derived from public opinions on Twitter. It also shows how different machine learning models classify tweets into positive and negative categories with the same available data.

## How To Run Locally
1. Clone this repository to your local machine.
2. Navigate to the `bar_graph_visualization` directory within the cloned repo through your command-line interface.
3. Run the following command to start up a local server:
    ```bash
    python3 -m http.server 8000
    ```
    > This command will start up a server at http://localhost:8000. 
4. To view the bar graph visualization, paste the following link into a web browser of your choosing: http://localhost:8000/bar_graph_visualization/bar_graphs.html

## Functionality Details
This interface allows for deeper analysis through selecting specific teams and viewing their sentiment levels.

For the selected team, information is displayed for the following three classifiers, each with their own bar graph:
- Random Forest 
- Naive Bayes
- Linear Regression

### Bar Graph Elements:
- Team Select
  - This option is where you can select the teams you specifically want to visualize on the graphs. 
- Interactive Tool Tip
	- Hover your mouse over a bar within a graph to view the corresponding sentiments and number of tweets for this category.

