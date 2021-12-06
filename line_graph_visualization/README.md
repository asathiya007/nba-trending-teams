# NBA Trending Teams (Tweet-Based Line Graph Visualization)

## Overview
This visualization shows an overall sentiment trend for each of the NBA teams over a specified period of time.

Our trained models were used to classify the sentiment for each of the Tweets as either positive or negative. Based on this, each of the NBA teams has a positive Tweets count and a negative Tweets count for every day that data was collected.

The nature of this data led to two graphs being displayed on the HTML page: a positive Tweet counts vs. date graph and a negative Tweet counts vs. date graph.

Note: the line graph visualization data is a processed version of the model output data. The `data_manipulation.ipynb` file contains logic for getting daily Tweet counts for each team and getting sample Tweets for each team on each day.

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
