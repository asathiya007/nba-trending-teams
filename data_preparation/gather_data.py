import tweepy 

consumer_key = 'MB6nEjXaf7SGnoODGrOPkfqKW'
consumer_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIqdVAEAAAAAbV6%2BKw11NqoCCKoaVIs9q7M07SQ%3DyOahpZv3b0bqqMK186y0XMPSTD1hjM4plBALNCLBEUgFE7JSh2'
access_token = 'MB6nEjXaf7SGnoODGrOPkfqKW'
access_token_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'

client = tweepy.Client(bearer_token, consumer_key, consumer_secret, access_token, 
    access_token_secret, wait_on_rate_limit=True)

query = 'warriors OR dubnation'
max_results = 10 
expansions = 'author_id'
user_fields = 'location'
tweets = client.search_recent_tweets(query=query, max_results=max_results, 
    expansions=expansions, user_fields=user_fields)

# to get tweet text 
# for tweet in tweets.data: 
#     print('Author ID: ', tweet.author_id)
#     print('Tweet Text: ', tweet.text)
#     print('\n----\n')

user = client.get_user(id='485441229', user_fields='username,location')
print(dir(user))
print(user)
'''
when the same request is made from Postman, this is what is returned: 

{
    "data": {
        "username": "kiiru_gichora",
        "id": "485441229",
        "name": "YOUNG the GIANT",
        "location": "Nairobi,KE"
    }
}

this is what is printed in the terminal: 
Response(data=<User id=485441229 name=YOUNG the GIANT username=kiiru_gichora>, includes={}, errors=[], meta={})

no location data...
'''