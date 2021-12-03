# NBA Team Sentiment Visualizer (Tweet-based Bar Graph)

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

