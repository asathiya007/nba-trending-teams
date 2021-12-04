import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from data import load_count_vectorizer, load_tfidf_transformer, get_clean_tokens
from models import load_logistic_regression, load_naive_bayes, load_random_forest
import time 

print('Started predicting sentiments of collection of recent NBA tweets')
start = time.time()

# load tweets into dataframe
df = pd.read_csv("../NBAData.csv")

rf_predictions_pos = []
rf_predictions_neg = []
nb_predictions_pos = []
nb_predictions_neg = []
lr_predictions_pos = []
lr_predictions_neg = []

# load stemmer and vectorizers 
snowball_stemmer = SnowballStemmer(language='english')
count_vectorizer = load_count_vectorizer()
tfidf_transformer = load_tfidf_transformer() 
naive_bayes = load_naive_bayes() 
logistic_regression = load_logistic_regression()
random_forest = load_random_forest() 

# find predictions from each model
for index, d in df.iterrows():
    if index % 100 == 0:
        print(index)

    # transform tweet 
    tweet = d['Tweet']
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

    rf_prediction = random_forest.predict_proba(tweet_tfidf)[0]
    nb_prediction = naive_bayes.predict_proba(tweet_tfidf)[0]
    lr_prediction = logistic_regression.predict_proba(tweet_tfidf)[0]
    
    rf_predictions_pos.append(rf_prediction[1])
    rf_predictions_neg.append(rf_prediction[0])
    nb_predictions_pos.append(nb_prediction[1])
    nb_predictions_neg.append(nb_prediction[0])
    lr_predictions_pos.append(lr_prediction[1])
    lr_predictions_neg.append(lr_prediction[0])

# append prediction probabilities into dataframe
df["RF Positive"] = rf_predictions_pos
df["RF Negative"] = rf_predictions_neg
df["NB Positive"] = nb_predictions_pos
df["NB Negative"] = nb_predictions_neg
df["LR Positive"] = lr_predictions_pos
df["LR Negative"] = lr_predictions_neg

# write to csv
df.to_csv("classifiedNBAData.csv")

# report total time 
end = time.time()
print('Finished predicting sentiments of collection of recent NBA tweets')
print('Total time for all predictions: ', end - start, 
    ' seconds\n')