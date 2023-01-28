import pandas as pd
import codecs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st 

st.title('SMS SPAM DETECTION')

with codecs.open('spam.csv','r',encoding='utf-8',errors='ignore') as f:
    dataset=pd.read_csv(f)

dataset=dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
dataset=dataset.rename(columns={'v1':'label','v2':'msg'})
dataset['length']=dataset['msg'].apply(len)
dataset['output']=dataset.label.map({'ham':0,'spam':1})

x=dataset.msg
x=x.values

y=dataset.output
y=y.values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

vect=CountVectorizer()
x_train=vect.fit_transform(x_train)
x_test=vect.transform(x_test)

classifier1=LogisticRegression(solver='liblinear',penalty='l1')
classifier1.fit(x_train,y_train)

k=st.text_input('Enter Message')
if st.button('Classify SMS'):
    data=[k]
    data=vect.transform(data)
    result=classifier1.predict(data)
    if result==0:
        st.success('Not A Spam')
    else:
        st.error('Spam Message')
        