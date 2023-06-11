import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk
import re

nltk.download("punkt")
nltk.download("stopwords")

tfidf= pickle.load(open("tfidf.pkl","rb"))
model= pickle.load(open("model.pkl","rb"))

st.title("Movie Sentiment Analysis")
text = st.text_area("Paste the comment below:")

ps = PorterStemmer()
pattern = re.compile('<.*?>')

def transform(text):
  
  text = (re.sub(pattern,'',text)).replace('\'','')
  text = text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

if st.button("Predict"):
    text = transform(text)
    text = tfidf.transform([text])
    
    prediction = model.predict(text)
    
    if prediction[0] == 0:
        st.success("It is a Negative Comment.")
        
    else:
        st.success("It is a Positive Comment.")
    