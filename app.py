import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Surefy.AI")
st.title("SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    
    transformed_sms = transform_text(input_sms)
    
    vector_input = tfidf.transform([transformed_sms])
    
    result = model.predict(vector_input)[0]
    
    if result == 3:
        st.header("Not Spam")
    elif result== 4:
        st.header("Spam")
    elif result== 2:
        st.header("OTP")
    elif result== 0:
        st.header("Fraud")
    elif result== 1:
        st.header("Logistic")

        if st.button("Track Order"):
            
            st.write("Redirecting to Order Tracking... ")
            
