# Twitter Sentiment Analysis with cardiffnlp/twitter-roberta-base-sentiment-latest Model

This Python-based code uses the ```twitter-roberta-base``` transformer model to analyze the sentiment of tweets from "The Telegraph" twitter channel. This model is a pre-trained transformer model that can classify tweets based on their sentiment.

### **Installation**
To use this code, you must have ```Python 3.x``` installed on your system. 

- You can download the required packages by running the following command:
```pip install -r requirements.txt```

- This will install all the required dependencies, including the Hugging Face transformers library and the snscrape package.
- Also make sure that ```PyTorch``` is installed on your system.

### **Usage**
To use this code, simply run the ```twitter_sentiment_analysis.py``` file by typing the following command:

- ```python twitter_sentiment_analysis.py```
This script will scrape the most recent tweets(Yesterday-Today) from "The Telegraph" Twitter channel using ```snscrape``` and analyze the sentiment of the tweets. The sentiment analysis results will be printed to the console.

### **References**
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [snscrape package](https://github.com/JustAnotherArchivist/snscrape)
