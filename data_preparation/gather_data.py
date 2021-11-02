import tweepy 
import json
consumer_key = 'MB6nEjXaf7SGnoODGrOPkfqKW'
consumer_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIqdVAEAAAAAbV6%2BKw11NqoCCKoaVIs9q7M07SQ%3DyOahpZv3b0bqqMK186y0XMPSTD1hjM4plBALNCLBEUgFE7JSh2'
access_token = 'MB6nEjXaf7SGnoODGrOPkfqKW'
access_token_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'

client = tweepy.Client(bearer_token, consumer_key, consumer_secret, access_token, 
    access_token_secret, wait_on_rate_limit=True)

query = 'warriors OR dubnation'
max_results = 100 #between 10 and 100
expansions = 'author_id'
user_fields = 'location'
tweets = client.search_recent_tweets(query=query, max_results=max_results, 
    expansions=expansions, user_fields=user_fields)

all_users = []
# to get tweet text 
for tweet in tweets.data: 
    # print('Author ID: ', tweet.author_id)
    # print('Tweet Text: ', tweet.text)
    # print('\n----\n')
    all_users.append(tweet.author_id)

#go through all the useridss of  tweets
#go  through the locations - figurer out whatt statess are theer
all_states = []
for i in all_users:
    user = client.get_user(id=i, user_fields='location')
    location = user.data.location
    all_states.append(location)
# print(all_states)

state_acronym = ["AL", "AK", "AZ", 'AR', 'CA', 'CO', 'CT',  'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND','OH', 'OK', 'OR', 'PA', 'RI', 'SC','TN','TX','UT','VT','VA','WA','WV','WI','WY', 'DC']
state_name =["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

all_locations = set()
for loc in all_states:
    for s in range(50):
        if loc:
            if state_acronym[s] in loc or state_name[s].lower() in loc.lower():
                all_locations.add(state_name[s])
print(all_locations)


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