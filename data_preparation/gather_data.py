import tweepy 
import json

consumer_key = 'MB6nEjXaf7SGnoODGrOPkfqKW'
consumer_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIqdVAEAAAAAbV6%2BKw11NqoCCKoaVIs9q7M07SQ%3DyOahpZv3b0bqqMK186y0XMPSTD1hjM4plBALNCLBEUgFE7JSh2'
access_token = 'MB6nEjXaf7SGnoODGrOPkfqKW'
access_token_secret = 'cXKDXCnx8NmBkN2XKVDYdswXL9CYWpPdw48BAsf1NOFznfHKN6'

client = tweepy.Client(bearer_token, consumer_key, consumer_secret, access_token, 
    access_token_secret, wait_on_rate_limit=False)
state_acronym = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
state_name =["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
all_teams = ["celtics", "nets", "knicks", "76ers", "raptors", "bulls", "cavaliers", "pistons", "pacers", "bucks", "hawks", "hornets", "heat", "magic", "wizards", "nuggets", "timberwolves", "thunder", "blazers", "jazz", "lakers", "suns", "kings", "mavericks", "rockets", "grizzlies", "pelicans", "spurs"]
final_data = {}
for s in state_name:
    final_data[s] = {}
    for t in all_teams:
        final_data[s][t] = []
        
count = 0
for team in all_teams:
    query = team#'warriors OR dubnation'
    max_results = 30 #between 10 and 100 https://developer.twitter.com/en/docs/twitter-api/rate-limits
    expansions = 'author_id'
    user_fields = 'location'
    tweets = client.search_recent_tweets(query=query, max_results=max_results, 
        expansions=expansions, user_fields=user_fields)
    # print(tweets)
    

    all_users = {} #dict of users to tweets
    # to get tweet text 
    for tweet in tweets.data: 
        # print('Author ID: ', tweet.author_id)
        # print('Tweet Text: ', tweet.text)
        # print('\n----\n')


        if tweet.author_id not in all_users:
            all_users[tweet.author_id] = [tweet.text] #tweet.text
        else:
            all_users[tweet.author_id].append(tweet.text)

    # print(all_users)
    #go through all the useridss of  tweets
    #go  through the locations - figurer out whatt statess are theer
    all_tweets = {} #location : tweets
    for i in all_users:
        user = client.get_user(id=i, user_fields='location')
        location = user.data.location
        for s in range(50):
            if location:
                if state_acronym[s] in location or state_name[s].lower() in location.lower():
                    state = state_name[s]
                    if state not in all_tweets:
                        all_tweets[state] = all_users[i]
                    else:
                        all_tweets[state] += all_users[i]
    for s in all_tweets:
        count += len(all_tweets[s])
        final_data[s][team] += all_tweets[s]
    
print(final_data)
print(count)


        
# print(all_states)

# all_locations = set()
# for loc in all_states:
#     for s in range(50):
#         if loc:
#             if state_acronym[s] in loc or state_name[s].lower() in loc.lower():
#                 all_locations.add(state_name[s])
# print(all_locations)


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