import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import sklearn
# A function that will perform all these processes
def transform_text(text):
    text=text.lower() #converting text into lower alphabets
    text=nltk.word_tokenize(text) #tokenisng text
    
    #removing special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    #removing stopwords and punctuation
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    #stemming
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_area("Enter the message : ")
if st.button('Predict'):

    #1. preprocess
    transformed_sms=transform_text(input_sms)
    #2. Vectorize
    vector_input=tfidf.transform([transformed_sms])
    #3. predict
    #model.predict(vector_input)[0] returns the predicted class label for the input SMS message after it has been preprocessed, vectorized, and passed through the machine learning model
    result=model.predict(vector_input)[0]
    #4. Display
    if result==1:
        st.header("SPAM!!")
    else:
        st.header("NOT SPAM :)")
