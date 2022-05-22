from SENTIMENT_ANALYSIS import *

obj=Sentiment()
obj.load_sentiment()
text = input("Review: ")
if obj.classify_sentiment(text):
    print("This review is about ",obj.classify_sentiment(text))
print("and this is a ",obj.predict_sentiment(text),"review.")