#Imports
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

import snscrape.modules.twitter as sntwitter 
#Twitter tweet scrapper.Can also be used to scrape from other social platforms

import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta


# Pre-processing the tweets to remove username and introduce link placeholders

def preprocess(tweet):
    new_tweet = []
    for char in tweet.split(" "):
        if char.startswith("@") and len(char)>1:
            char = "@user"
        elif char.startswith("http"):
            char = "http"
        new_tweet.append(char)
    return " ".join(new_tweet)

# Twitter news sentiment analysis
today = date.today()
yesterday = date.today() - timedelta(days=1)

twitter_channel = "Telegraph"
#query = "Elon Musk"
#query = "(from:Telegraph) lang:en"
#query = "(from:Telegraph) lang:en until:2023-04-10 since:2023-04-03"

query = f"(from:{twitter_channel}) min_faves:50 lang:en until:{today} since:{yesterday}"
tweets= []
limit = 100
         

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # print(vars(tweet).keys())
    # break
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

print('The created dataframe is:')
print(df)



# Defining our transformer model for classification
model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

# Will use tokenizer of the mentioned model to convert tweets to tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)


################################ Analyzing tweet sentiment #################################

print('The inidividual tweets are:')
f=1
for tweet in df['Tweet']:
    
    if f==11:
        break
    
    print(f'Sentence {f} is: {tweet} \n and associated sentiment of the tweet is')
    tweet = preprocess(tweet)
    
    encoded_tweet = tokenizer(tweet, return_tensors="pt")
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    
    f+=1






