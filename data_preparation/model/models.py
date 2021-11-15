from data import process_data, transform_tweet
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import time 
from zipfile import ZipFile


def _save_random_forest(random_forest): 
    # save random forest model 
    joblib.dump(random_forest, './random_forest.joblib')

def _load_random_forest(): 
    # load random forest model 
    random_forest = joblib.load('./random_forest.joblib')

    # return loaded model 
    return random_forest 

def _fit_random_forest(x_train, x_test, y_train, y_test): 
    # fit random forest model 
    random_forest = RandomForestClassifier(n_estimators=25, max_depth=750)
    random_forest.fit(x_train, y_train)

    # evaluate random forest model 
    preds = random_forest.predict(x_test)
    f1 = f1_score(y_test, preds) 
    print('Random forest F1 score: ', f1)
    accuracy = accuracy_score(y_test, preds)
    print('Random forest accuracy: ', accuracy)

    # save random forest model 
    _save_random_forest(random_forest)

def predict_random_forest(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load random forest model 
    random_forest = _load_random_forest()

    # predict sentiment of tweet 
    pred = random_forest.predict_proba(tweet)

    # return prediction 
    return pred[0]

def _save_naive_bayes(naive_bayes): 
    # save naive Bayes model 
    joblib.dump(naive_bayes, './naive_bayes.joblib')

def _load_naive_bayes(): 
    # load naive Bayes model 
    naive_bayes = joblib.load('./naive_bayes.joblib')

    # return loaded model 
    return naive_bayes 

def _fit_naive_bayes(x_train, x_test, y_train, y_test): 
    # fit naive Bayes model 
    naive_bayes = MultinomialNB() 
    naive_bayes.fit(x_train, y_train)

    # evaluate naive Bayes model 
    preds = naive_bayes.predict(x_test)
    f1 = f1_score(y_test, preds) 
    print('Naive Bayes F1 score: ', f1)
    accuracy = accuracy_score(y_test, preds)
    print('Naive Bayes accuracy: ', accuracy)

    # save naive Bayes model 
    _save_naive_bayes(naive_bayes)

def predict_naive_bayes(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load naive Bayes model 
    naive_bayes = _load_naive_bayes()

    # predict sentiment of tweet 
    pred = naive_bayes.predict_proba(tweet)

    # return prediction 
    return pred[0]

def _save_linear_regression(linear_regression): 
    # save linear regression model 
    joblib.dump(linear_regression, './linear_regression.joblib')

def _load_linear_regression(): 
    # load linear regression model 
    linear_regression = joblib.load('./linear_regression.joblib')

    # return loaded model 
    return linear_regression 

def _fit_linear_regression(x_train, x_test, y_train, y_test): 
    # fit linear regression model 
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    # evaluate linear regression model 
    preds = linear_regression.predict(x_test)
    for i in range(len(preds)):
        if preds[i] >= 0.5: 
            preds[i] = 1
        else: 
            preds[i] = 0
    f1 = f1_score(y_test, preds) 
    print('Linear regression F1 score: ', f1)
    accuracy = accuracy_score(y_test, preds)
    print('Linear regression accuracy: ', accuracy)

    # save linear regression model 
    _save_linear_regression(linear_regression)

def predict_linear_regression(tweet): 
    # transform tweet 
    tweet = transform_tweet(tweet)

    # load linear regression model 
    linear_regression = _load_linear_regression()

    # predict sentiment of tweet 
    pred = linear_regression.predict(tweet)      

    # return prediction 
    return pred[0]

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
    _fit_naive_bayes(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Finished training naive Bayes model')
    print('Total time for training naive Bayes model: ', end - start, 
        ' seconds\n')

    # fit linear regression
    print('Started training linear regression model')
    start = time.time()
    _fit_linear_regression(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Finished training linear regression model')
    print('Total time for training linear_regression: ', end - start, 
        ' seconds\n')

    # fit random forest 
    print('Started training random forest model')
    start = time.time()
    _fit_random_forest(x_train, x_test, y_train, y_test)
    end = time.time()
    print('Finished training random forest model')
    print('Total time for training random forest: ', end - start, 
        ' seconds\n')

    # zip files 
    file_names = [
        './count_vectorizer.pkl', 
        './tfidf_transformer.pkl', 
        './naive_bayes.joblib',
        './linear_regression.joblib', 
        './random_forest.joblib'
    ]
    zip_file = ZipFile('./ml_resources.zip', 'w')
    for file_name in file_names: 
        zip_file.write(file_name)
    zip_file.close()
    print('Saved machine learning resources in ./ml_resources.zip\n')

if __name__ == '__main__':
    fit_models()
