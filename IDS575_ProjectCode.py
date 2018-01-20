import csv
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


#Method to read the CSV and only pick the reviewText column
def readFile(filename):
	flat_list=[]
	with open(filename,'r') as f:
		reader=csv.reader(f)
		reviews=list(reader)
		for sublist in reviews:
		    for item in sublist:
		        flat_list.append(item)
	return flat_list
		
r=readFile('sampleReviews.csv')
#print(r)

df = pd.read_csv('sampleReviews.csv', header =None)
df.columns=["Reviews","Score"]
#print(df.head(10))
df = shuffle(df)
#print(df.head(10))

train, test = train_test_split(df, test_size=0.3)

#print(len(train))

tokenize = lambda doc: doc.lower().split(" ")

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
trainMatrix = sklearn_tfidf.fit_transform(train['Reviews'])
#print(sklearn_representation)
testMatrix = sklearn_tfidf.transform(test['Reviews'])

D = LogisticRegression()
D = D.fit(trainMatrix,train['Score'])
pred = D.predict(testMatrix)
pred2 = pred.astype(int)
print(classification_report(test['Score'], pred2))
print("Type for score"+str(test['Score'].dtype))
print("Type for pred"+str(type(pred)))


print(test['Score'])

features=sklearn_tfidf.get_feature_names()
trainMatrix=trainMatrix.toarray()

matrix_intercept = np.append(trainMatrix,np.ones([len(trainMatrix),1]),1)

coeff=np.asarray([0.0 for i in range(len(matrix_intercept[0]))])

def costfunc(X,y,coeff):
	J = np.sum((X.dot(coeff)-y)**2)/2/5000
	return J

def regression(X, y, coef, lr=0.05, n_epoch=200):
	ch=[0]*n_epoch
	#print(X.dtype)
	y=y.astype(int)
	for i in range(n_epoch):
		hyp=X.dot(coef)
		loss=np.array(hyp) - y
		gradient=X.T.dot(loss)/5000
		coef=coef-lr*gradient
		cost=costfunc(X,y,coef)
		ch[i]=cost 
	return coef,ch

(coefficients,error)=regression(matrix_intercept,train['Score'],coeff)
print("Coefficiets are:")
print(coefficients)
print("-"*30)
print("Errors in 200 iterations")
print(error)
plt.plot(error)
plt.ylabel("RMSE")
plt.xlabel("epochs")
plt.show()


