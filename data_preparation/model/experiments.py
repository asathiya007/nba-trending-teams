# import modules 
from data import (load_count_vectorizer, load_tfidf_transformer, 
    process_data)
from models import (load_logistic_regression, load_naive_bayes, 
    load_random_forest, fit_logistic_regression, fit_naive_bayes, 
    fit_random_forest, predict_multiple_logistic_regression, 
    predict_multiple_naive_bayes, predict_multiple_random_forest)
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem.snowball import SnowballStemmer
import statistics
import time


# define constants
ITERATIONS = 5


def test_data_processing_model_fitting(holdout): 
    # initialize lists 
    data_processing_times = [] 
    logistic_regression_fit_times = [] 
    logistic_regression_f1s = [] 
    logistic_regression_accuracies = [] 
    naive_bayes_fit_times = [] 
    naive_bayes_f1s = [] 
    naive_bayes_accuracies = [] 
    random_forest_fit_times = [] 
    random_forest_f1s = [] 
    random_forest_accuracies = [] 

    # run iterations and record results
    print('Testing data processing time and model fitting')
    for _ in range(ITERATIONS):

        # process data 
        start = time.time()
        x_train, x_test, y_train, y_test = process_data(holdout) 
        end = time.time()
        data_processing_times.append(end - start)

        # fit logistic regression model 
        start = time.time()
        _, f1, accuracy = fit_logistic_regression(x_train, x_test, y_train, y_test)
        end = time.time()
        logistic_regression_fit_times.append(end - start)
        logistic_regression_f1s.append(f1)
        logistic_regression_accuracies.append(accuracy)

        # fit naive bayes model 
        start = time.time()
        _, f1, accuracy = fit_naive_bayes(x_train, x_test, y_train, y_test)
        end = time.time()
        naive_bayes_fit_times.append(end - start)
        naive_bayes_f1s.append(f1)
        naive_bayes_accuracies.append(accuracy)

        # fit random forest model 
        start = time.time()
        _, f1, accuracy = fit_random_forest(x_train, x_test, y_train, y_test)
        end = time.time()
        random_forest_fit_times.append(end - start)
        random_forest_f1s.append(f1)
        random_forest_accuracies.append(accuracy)

    # compute additional statistics for each list 
    data = [
        data_processing_times, 
        logistic_regression_fit_times, 
        logistic_regression_f1s, 
        logistic_regression_accuracies, 
        naive_bayes_fit_times, 
        naive_bayes_f1s, 
        naive_bayes_accuracies, 
        random_forest_fit_times, 
        random_forest_f1s, 
        random_forest_accuracies
    ] 
    for i in range(len(data)): 
        data_list = data[i]
        mean = statistics.mean(data_list)
        median = statistics.median(data_list)
        rng = max(data_list) - min(data_list)
        variance = statistics.variance(data_list)
        stdev = statistics.stdev(data_list)
        data_list += [mean, median, rng, variance, stdev]
        data[i] = data_list 

    # save result as CSV 
    index = [
        'data processing time (s)',
        'logistic regression fit time (s)', 
        'logistic regression f1',
        'logistic regression accuracy',
        'naive Bayes fit time (s)', 
        'naive Bayes f1',
        'naive Bayes accuracy',
        'random forest fit time (s)', 
        'random forest f1',
        'random forest accuracy'
    ]
    columns = ['iteration_' + str(i + 1) for i in range(ITERATIONS)] + [
        'mean', 'median', 'range', 'variance', 'stdev']
    data_processing_model_fitting = pd.DataFrame(data, index, columns)
    data_processing_model_fitting.to_csv(
        f'./data_processing_model_fitting_holdout={holdout}.csv')

def test_load_times(holdout):
    # test load times for vectorizers
    print('Testing load time for stemmer and vectorizers')
    stemmer_vectorizer_load_times = [] 
    for _ in range(ITERATIONS):
        start = time.time() 
        _ = SnowballStemmer(language='english')
        _ = load_count_vectorizer() 
        _ = load_tfidf_transformer()
        end = time.time() 
        stemmer_vectorizer_load_times.append(end - start)
    mean = statistics.mean(stemmer_vectorizer_load_times)
    median = statistics.median(stemmer_vectorizer_load_times)
    rng = max(stemmer_vectorizer_load_times) - min(
        stemmer_vectorizer_load_times)
    variance = statistics.variance(stemmer_vectorizer_load_times)
    stdev = statistics.stdev(stemmer_vectorizer_load_times)
    stemmer_vectorizer_load_times += [mean, median, rng, variance, stdev]
    
    # test load times for logistic regression model 
    print('Testing load time for logistic regression model')
    logistic_regression_load_times = [] 
    for _ in range(ITERATIONS):
        start = time.time() 
        _ = load_logistic_regression() 
        end = time.time() 
        logistic_regression_load_times.append(end - start)
    mean = statistics.mean(logistic_regression_load_times)
    median = statistics.median(logistic_regression_load_times)
    rng = max(logistic_regression_load_times) - min(
        logistic_regression_load_times)
    variance = statistics.variance(logistic_regression_load_times)
    stdev = statistics.stdev(logistic_regression_load_times)
    logistic_regression_load_times += [mean, median, rng, variance, stdev]

    # test load times for naive bayes model 
    print('Testing load time for naive Bayes model')
    naive_bayes_load_times = [] 
    for _ in range(ITERATIONS):
        start = time.time() 
        _ = load_naive_bayes() 
        end = time.time() 
        naive_bayes_load_times.append(end - start)
    mean = statistics.mean(naive_bayes_load_times)
    median = statistics.median(naive_bayes_load_times)
    rng = max(naive_bayes_load_times) - min(naive_bayes_load_times)
    variance = statistics.variance(naive_bayes_load_times)
    stdev = statistics.stdev(naive_bayes_load_times)
    naive_bayes_load_times += [mean, median, rng, variance, stdev]

    # test load times for random forest model 
    print('Testing load time for random forest model')
    random_forest_load_times = [] 
    for _ in range(ITERATIONS):
        start = time.time() 
        _ = load_random_forest() 
        end = time.time() 
        random_forest_load_times.append(end - start)
    mean = statistics.mean(random_forest_load_times)
    median = statistics.median(random_forest_load_times)
    rng = max(random_forest_load_times) - min(random_forest_load_times)
    variance = statistics.variance(random_forest_load_times)
    stdev = statistics.stdev(random_forest_load_times)
    random_forest_load_times += [mean, median, rng, variance, stdev]
    
    # save result as CSV 
    index = [
        'stemmer and vectorizers load time (s)', 
        'logistic regression load time (s)', 
        'naive Bayes load time (s)', 
        'random forest load time (s)'
    ]
    columns = ['iteration_' + str(i + 1) for i in range(ITERATIONS)] + [
        'mean', 'median', 'range', 'variance', 'stdev']
    data = [stemmer_vectorizer_load_times, logistic_regression_load_times, 
        naive_bayes_load_times, random_forest_load_times]
    load_times = pd.DataFrame(data, index, columns)
    load_times.to_csv(f'./load_times_holdout={holdout}.csv')

def test_predictions_NBA_tweets(holdout): 
    # load data from csv file 
    nba_tweets = pd.read_csv('../labeled_partitioned.csv')
    tweets = nba_tweets['tweet']
    labels = nba_tweets['label']

    # make predictions with logistic regression model 
    preds, stats = predict_multiple_logistic_regression(tweets)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    logistic_regression_stats = [accuracy, f1] + stats

    # make predictions with naive Bayes model 
    preds, stats = predict_multiple_naive_bayes(tweets)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    naive_bayes_stats = [accuracy, f1] + stats

    # make predictions with random forest model 
    preds, stats = predict_multiple_random_forest(tweets)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    random_forest_stats = [accuracy, f1] + stats

    # save result as CSV 
    index = [
        'logistic regression', 
        'naive Bayes', 
        'random forest'
    ]
    columns = [
        'accuracy',
        'f1',
        'mean_prediction_time', 
        'median_prediction_time', 
        'range_prediction_time', 
        'variance_prediction_time', 
        'stdev_prediction_time'
    ]
    data = [logistic_regression_stats, naive_bayes_stats, 
        random_forest_stats]
    load_times = pd.DataFrame(data, index, columns)
    load_times.to_csv(f'./prediction_stats_holdout={holdout}.csv')


if __name__=='__main__':
    # run experiments
    holdouts = [0.2, 0.3, 0.4]
    for holdout in holdouts: 
        test_data_processing_model_fitting(holdout)
        test_load_times(holdout)
        test_predictions_NBA_tweets(holdout)
