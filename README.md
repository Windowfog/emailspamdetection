import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data=pd.read_csv('spam.csv')
data

data.columns

data.info()

data.isna().sum

data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data.Message, data.Spam, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
clf=Pipeline([('vectorizer', CountVectorizer()),('nb', MultinomialNB())])

clf.fit(x_train, y_train)

emails=['sounds great! are you home now?', 'Will u meet your dream partner soon? Is ur career off 2 a flying start? 2 find out free, txt HORO followed by ur star sign, e.g. HORO ARIES']

# predict email
clf.predict(emails)

# prediction of model
clf.score(x_test, y_test)
