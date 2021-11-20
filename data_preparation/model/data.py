# import modules 
import nltk 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


def _get_data():
    # read data from CSV file 
    data = pd.read_csv('./sentiment140dataset.csv', header=None)

    # extract tweet text and labels 
    data.rename(columns={0: 'label', 1: 'id', 2: 'date', 3: 'query', 
        4: 'user', 5: 'tweet'}, inplace=True)
    dataset = data[['label', 'tweet']].copy(deep=True)
    
    # adjust labels 
    dataset.loc[dataset['label'] == 4, 'label'] = 1
    
    # return dataset 
    return dataset 

def get_clean_tokens(tweet): 
    # tokenize tweet 
    tokens = tweet.split() 

    # clean each token 
    clean_tokens =[]
    for token in tokens: 
        token = token.strip() 

        # remove mentions and media
        if len(token) == 0 or token[0] == '@' or '://' in token: 
            continue
        extensions = ['.com', '.org', '.net']
        extension_found = False
        for extension in extensions: 
            if extension in token: 
                extension_found = True 
        if extension_found: 
            continue 

        # remove non-alphaneumeric characters
        regex = re.compile('[^a-zA-Z0-9]')
        token = regex.sub('', token)

        # record cleaned token
        if len(token) != 0: 
            clean_tokens.append(token)

    # return list of clean tokens 
    return clean_tokens
    
def _tokenize_tweets(dataset): 
    # get clean tokens of tweet
    dataset['tweet'] = dataset['tweet'].apply(
        lambda tweet: get_clean_tokens(tweet))
    
    # return dataset 
    return dataset 

def _normalize_tweets(dataset): 
    # normalize tweet text using stemming 
    stemmer = SnowballStemmer('english')
    dataset['tweet'] = dataset['tweet'].apply(lambda tokens: 
        [stemmer.stem(token) for token in tokens])
    
    # return dataset 
    return dataset

def _remove_stopwords_from_tweets(dataset):
    # remove stopwords
    nltk.download('stopwords')
    eng_stopwords = stopwords.words('english')
    dataset['tweet'] = dataset['tweet'].apply(lambda tokens: 
        [token for token in tokens if token not in eng_stopwords])
    
    # return dataset 
    return dataset 

def _vectorize_tweets(x_train, x_test): 
    # vectorize using counts 
    count_vectorizer = CountVectorizer(stop_words='english', 
        max_features=10000)
    x_train_counts = count_vectorizer.fit_transform(x_train)
    x_test_counts = count_vectorizer.transform(x_test)
    
    # vectorize from counts using tf-idf 
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    
    # return dataset 
    return x_train_tfidf, x_test_tfidf, count_vectorizer, tfidf_transformer

def _save_count_vectorizer(count_vectorizer): 
    # save count vectorizer 
    with open('./count_vectorizer.pkl', 'wb') as f: 
        pickle.dump(count_vectorizer, f)

def load_count_vectorizer(): 
    # load count vectorizer 
    with open('./count_vectorizer.pkl', 'rb') as f: 
        count_vectorizer = pickle.load(f)
    
    # return count vectorizer 
    return count_vectorizer

def _save_tfidf_transformer(tfidf_transformer): 
    # save count vectorizer 
    with open('./tfidf_transformer.pkl', 'wb') as f: 
        pickle.dump(tfidf_transformer, f)

def load_tfidf_transformer(): 
    # load count vectorizer 
    with open('./tfidf_transformer.pkl', 'rb') as f: 
        tfidf_transformer = pickle.load(f)
    
    # return count vectorizer 
    return tfidf_transformer

def process_data(holdout=0.2): 
    # get data from file 
    dataset = _get_data() 
    
    # tokenize tweets
    dataset = _tokenize_tweets(dataset)
    
    # normalize tweet text 
    dataset = _normalize_tweets(dataset)
    
    # remove stopwords 
    dataset = _remove_stopwords_from_tweets(dataset)
    
    # join tweets, split data into train and test sets 
    dataset['tweet'] = dataset['tweet'].apply(lambda tokens: ' '.join(tokens))
    x_train, x_test, y_train, y_test = train_test_split(dataset['tweet'], 
        dataset['label'], test_size=holdout, shuffle=True) 
    
    # vectorize features
    x_train, x_test, count_vectorizer, tfidf_transformer = _vectorize_tweets(
        x_train, x_test)

    # save count vectorizer 
    _save_count_vectorizer(count_vectorizer)

    # save tfidf transformer 
    _save_tfidf_transformer(tfidf_transformer)
    
    # return final dataset 
    return x_train, x_test, y_train, y_test

def transform_tweet(tweet): 
    # load stemmer, vectorizers and stopwords 
    try: 
        eng_stopwords = stopwords.words('english')
    except: 
        nltk.download('stopwords')
        eng_stopwords = stopwords.words('english')
    snowball_stemmer = SnowballStemmer(language='english')
    count_vectorizer = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer() 

    # preprocess tweet 
    tokens = get_clean_tokens(tweet)
    new_tokens = []
    for token in tokens: 
        new_tokens.append(snowball_stemmer.stem(token)) 
    tokens = new_tokens 
    tokens = list(filter(lambda token: token not in eng_stopwords, tokens))

    # transform tweet 
    tweet_counts = count_vectorizer.transform(pd.DataFrame([tweet])[0])
    tweet_tfidf = tfidf_transformer.transform(tweet_counts)

    # return transformed tweet 
    return tweet_tfidf