import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model

class Sentiment:
    def __init__(self):
        self.df=pd.read_csv("K:\SENTIMENT\Tweets.csv\Tweets.csv")
        self.review_df = self.df[['text','airline_sentiment']]
        self.review_df = self.review_df[self.review_df['airline_sentiment'] != 'neutral']
        self.sentiment_label = self.review_df.airline_sentiment.factorize()
        self.tweet = self.review_df.text.values
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(self.tweet)
        self.encoded_docs = self.tokenizer.texts_to_sequences(self.tweet)
        self.padded_sequence = pad_sequences(self.encoded_docs, maxlen=200)
        self.embedding_vector_length = 32
        self.model = Sequential()

    def train_sentiment(self):
        self.model.add(Embedding(9900, self.embedding_vector_length, input_length=200))
        self.model.add(SpatialDropout1D(0.25))
        self.model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.padded_sequence,self.sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

    def summary_sentiment(self):
        print(self.model.summary())

    def predict_sentiment(self,text):
        text=re.sub(r'@[A-Za-z0-9-_]+', "",text)
        tw = self.tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw,maxlen=200)
        prediction = int(self.model.predict(tw).round().item())
        return self.sentiment_label[1][prediction]

    def classify_sentiment(self,text):
        try:
            return re.sub(r'@', "", re.search(r'@[A-Za-z0-9-_]+', text).group())
        except:
            return False

    def save_sentiment(self):
        self.model.save("K:\SENTIMENT\setimentor.model", save_format="h5")
    
    def load_sentiment(self):
        self.model=load_model("K:\SENTIMENT\setimentor.model")

if __name__=="__main__"   :
    obj=Sentiment()
    # obj.train_sentiment()
    # obj.save_sentiment()
    obj.load_sentiment()
    test_sentence1 = "My mood is not good today"
    obj.predict_sentiment(test_sentence1)

    test_sentence2 = "You are bad boy"
    obj.predict_sentiment(test_sentence2)