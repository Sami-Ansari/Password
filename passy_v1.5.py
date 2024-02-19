import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
import xgboost as xgb
import random
import dill
import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_csv('data1.csv', on_bad_lines = "skip" )
df2 = pd.read_csv('data2.csv' , on_bad_lines = "skip" )
df3 = pd.read_csv('data3.csv', on_bad_lines = "skip" )
df4 = pd.read_csv('data4.csv', on_bad_lines = "skip" )
df5 = pd.read_csv('data5.csv', on_bad_lines = "skip" )
df6 = pd.read_csv('data6.csv', on_bad_lines = "skip" )
df7 = pd.read_csv('data7.csv', on_bad_lines = "skip" )
df8 = pd.read_csv('data8.csv', on_bad_lines = "skip" )
df9 = pd.read_csv('data9.csv' , on_bad_lines = "skip" )
df10 = pd.read_csv('data10.csv', on_bad_lines = "skip" )
df11 = pd.read_csv('data11.csv', on_bad_lines = "skip" )
df12 = pd.read_csv('data12.csv', on_bad_lines = "skip" )
df13 = pd.read_csv('data13.csv', on_bad_lines = "skip" )
df14 = pd.read_csv('data14.csv', on_bad_lines = "skip" )
df15 = pd.read_csv('data15.csv', on_bad_lines = "skip" )

frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15]

mydataset = pd.concat(frames)
print(mydataset)

mydataset.dropna(inplace=True)

password=np.array(mydataset)
random.shuffle(password)

X=[passwords[0] for passwords in password]
y=[passwords[1] for passwords in password]

def make_chars(inputs):
    characters=[]
    for letter in inputs:
        characters.append(letter)
    return characters

vectorizer=TfidfVectorizer(tokenizer=make_chars)
X_=vectorizer.fit_transform(X)
first_=X_[0].T.todense()

vec=pd.DataFrame(first_,index=vectorizer.get_feature_names_out(),columns=['tfidf'])
vec.sort_values(by=['tfidf'],ascending=False)

x_train,x_test,y_train,y_test=train_test_split(X_,y,test_size=0.27,random_state=42)

xgb_classifier=xgb.XGBClassifier(n_jobs=-1)
xgb_classifier.fit(x_train,y_train)
pred=xgb_classifier.predict(x_test)

confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))

model_file=open("xgb_classifier.pkl","wb")
dill.dump(xgb_classifier,model_file)
model_file.close()
dill.dump(vectorizer, open("vectorizer.pkl", "wb"))

password="musab@1234"
password=vectorizer.transform([password])
print(xgb_classifier.predict(password))





