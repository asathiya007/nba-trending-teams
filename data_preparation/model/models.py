from data import (process_data, transform_tweet, 
    load_count_vectorizer, load_tfidf_transformer, get_clean_tokens)
import joblib 
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import statistics 
import time 


def _save_random_forest(random_forest): 
    # save random forest model 
    joblib.dump(random_forest, './random_forest.joblib')

def load_random_forest(): 
    # load random forest model 
    random_forest = joblib.load('./random_forest.joblib')

    # return loaded model 
    return random_forest 

def fit_random_forest(x_train, x_test, y_train, y_test): 
    # fit random forest model 
    random_forest = RandomForestClassifier(n_estimators=10, max_depth=100)
    random_forest.fit(x_train, y_train)

    # evaluate random forest model 
    preds = random_forest.predict(x_test)
    f1 = f1_score(y_test, preds) 
    accuracy = accuracy_score(y_test, preds)

    # save random forest model 
    _save_random_forest(random_forest)

    # return model results 
    return random_forest, f1, accuracy

def predict_random_forest(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load random forest model 
    random_forest = load_random_forest()

    # predict sentiment of tweet 
    pred = random_forest.predict_proba(tweet)

    # return prediction 
    return pred[0]

def predict_multiple_random_forest(tweets, prob=False):
    # load stemmer, vectorizers, and model
    snowball_stemmer = SnowballStemmer(language='english')
    count_vectorizer = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer() 
    random_forest = load_random_forest() 

    # make predictions
    preds = [] 
    pred_times = [] 
    for tweet in tweets: 
        start = time.time() 
        # transform tweet
        try: 
            eng_stopwords = stopwords.words('english')
        except: 
            nltk.download('stopwords')
            eng_stopwords = stopwords.words('english')
        tokens = get_clean_tokens(tweet)
        new_tokens = []
        for token in tokens: 
            new_tokens.append(snowball_stemmer.stem(token)) 
        tokens = new_tokens 
        tokens = list(filter(lambda token: token not in eng_stopwords, tokens))
        tweet_counts = count_vectorizer.transform(pd.DataFrame([tweet])[0])
        tweet_tfidf = tfidf_transformer.transform(tweet_counts)

        # make prediction on tweet 
        if prob: 
            preds.append(random_forest.predict_proba(tweet_tfidf)[0])
        else: 
            preds.append(random_forest.predict(tweet_tfidf)[0])
        end = time.time() 
        pred_times.append(end - start)

    # return predictions and stats
    mean = statistics.mean(pred_times)
    median = statistics.median(pred_times)
    range = max(pred_times) - min(pred_times)
    variance = statistics.variance(pred_times)
    stdev = statistics.stdev(pred_times)
    stats = [mean, median, range, variance, stdev]
    return preds, stats

def _save_naive_bayes(naive_bayes): 
    # save naive Bayes model 
    joblib.dump(naive_bayes, './naive_bayes.joblib')

def load_naive_bayes(): 
    # load naive Bayes model 
    naive_bayes = joblib.load('./naive_bayes.joblib')

    # return loaded model 
    return naive_bayes 

def fit_naive_bayes(x_train, x_test, y_train, y_test): 
    # fit naive Bayes model 
    naive_bayes = MultinomialNB() 
    naive_bayes.fit(x_train, y_train)

    # evaluate naive Bayes model 
    preds = naive_bayes.predict(x_test)
    f1 = f1_score(y_test, preds) 
    accuracy = accuracy_score(y_test, preds)

    # save naive Bayes model 
    _save_naive_bayes(naive_bayes)

    # return model results 
    return naive_bayes, f1, accuracy

def predict_naive_bayes(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load naive Bayes model 
    naive_bayes = load_naive_bayes()

    # predict sentiment of tweet 
    pred = naive_bayes.predict_proba(tweet)

    # return prediction 
    return pred[0]

def predict_multiple_naive_bayes(tweets, prob=False):
    # load stemmer, vectorizers, and model
    snowball_stemmer = SnowballStemmer(language='english')
    count_vectorizer = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer() 
    naive_bayes = load_naive_bayes() 

    # make predictions
    preds = [] 
    pred_times = []
    for tweet in tweets: 
        # transform tweet
        start = time.time() 
        try: 
            eng_stopwords = stopwords.words('english')
        except: 
            nltk.download('stopwords')
            eng_stopwords = stopwords.words('english')
        tokens = get_clean_tokens(tweet)
        new_tokens = []
        for token in tokens: 
            new_tokens.append(snowball_stemmer.stem(token)) 
        tokens = new_tokens 
        tokens = list(filter(lambda token: token not in eng_stopwords, tokens))
        tweet_counts = count_vectorizer.transform(pd.DataFrame([tweet])[0])
        tweet_tfidf = tfidf_transformer.transform(tweet_counts)

        # make prediction on tweet 
        if prob: 
            preds.append(naive_bayes.predict_proba(tweet_tfidf)[0])
        else: 
            preds.append(naive_bayes.predict(tweet_tfidf)[0])
        end = time.time() 
        pred_times.append(end - start)

    # return predictions and stats
    mean = statistics.mean(pred_times)
    median = statistics.median(pred_times)
    range = max(pred_times) - min(pred_times)
    variance = statistics.variance(pred_times)
    stdev = statistics.stdev(pred_times)
    stats = [mean, median, range, variance, stdev]
    return preds, stats

def _save_logistic_regression(logistic_regression): 
    # save logistic regression model 
    joblib.dump(logistic_regression, './logistic_regression.joblib')

def load_logistic_regression(): 
    # load logistic regression model 
    logistic_regression = joblib.load('./logistic_regression.joblib')

    # return loaded model 
    return logistic_regression 

def fit_logistic_regression(x_train, x_test, y_train, y_test): 
    # fit logistic regression model 
    logistic_regression = LogisticRegression(max_iter=150)
    logistic_regression.fit(x_train, y_train)

    # evaluate logistic regression model 
    preds = logistic_regression.predict(x_test)
    f1 = f1_score(y_test, preds) 
    accuracy = accuracy_score(y_test, preds)

    # save logistic regression model 
    _save_logistic_regression(logistic_regression)

    # return model results 
    return logistic_regression, f1, accuracy

def predict_logistic_regression(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load logistic regression model 
    logistic_regression = load_logistic_regression()

    # predict sentiment of tweet 
    pred = logistic_regression.predict_proba(tweet)      

    # return prediction 
    return pred[0]

def predict_multiple_logistic_regression(tweets, prob=False):
    # load stemmer, vectorizers, and model
    snowball_stemmer = SnowballStemmer(language='english')
    count_vectorizer = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer() 
    logistic_regression = load_logistic_regression() 

    # make predictions
    preds = [] 
    pred_times = []
    for tweet in tweets: 
        # transform tweet
        start = time.time() 
        try: 
            eng_stopwords = stopwords.words('english')
        except: 
            nltk.download('stopwords')
            eng_stopwords = stopwords.words('english')
        tokens = get_clean_tokens(tweet)
        new_tokens = []
        for token in tokens: 
            new_tokens.append(snowball_stemmer.stem(token)) 
        tokens = new_tokens 
        tokens = list(filter(lambda token: token not in eng_stopwords, tokens))
        tweet_counts = count_vectorizer.transform(pd.DataFrame([tweet])[0])
        tweet_tfidf = tfidf_transformer.transform(tweet_counts)

        # make prediction on tweet 
        if prob: 
            preds.append(logistic_regression.predict_proba(tweet_tfidf)[0])
        else: 
            preds.append(logistic_regression.predict(tweet_tfidf)[0])
        end = time.time() 
        pred_times.append(end - start)

    # return predictions and stats
    mean = statistics.mean(pred_times)
    median = statistics.median(pred_times)
    range = max(pred_times) - min(pred_times)
    variance = statistics.variance(pred_times)
    stdev = statistics.stdev(pred_times)
    stats = [mean, median, range, variance, stdev]
    return preds, stats

def fit_models():
    # process data 
    print('Started data processing')
    start = time.time()
    x_train, x_test, y_train, y_test = process_data() 
    end = time.time()
    print('Finished data processing')
    print('Total time for data processing: ', end - start, ' seconds\n')

    # fit naive Bayes
    print('Started training naive Bayes model')
    start = time.time()
    _, f1, accuracy = fit_naive_bayes(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Naive Bayes F1 score: ', f1)
    print('Naive Bayes accuracy: ', accuracy)
    print('Finished training naive Bayes model')
    print('Total time for training naive Bayes model: ', end - start, 
        ' seconds\n')

    # fit logistic regression
    print('Started training logistic regression model')
    start = time.time()
    _, f1, accuracy = fit_logistic_regression(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Logistic regression F1 score: ', f1)
    print('Logistic regression accuracy: ', accuracy)
    print('Finished training logistic regression model')
    print('Total time for training logistic regression model: ', end - start, 
        ' seconds\n')

    # fit random forest 
    print('Started training random forest model')
    start = time.time()
    _, f1, accuracy = fit_random_forest(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Random forest F1 score: ', f1)
    print('Random forest accuracy: ', accuracy)
    print('Finished training random forest model')
    print('Total time for training random forest: ', end - start, 
        ' seconds\n')


if __name__ == '__main__':
    fit_models()
