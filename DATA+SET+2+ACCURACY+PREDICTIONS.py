
# coding: utf-8

# ## Importng the packages and modules required in the project

# In[328]:

import pandas as pd
import numpy as np
import csv
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.learning_curve import learning_curve
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger


# ## reading the data

# In[329]:

adv=pd.read_csv('fakerealnews.csv')


# In[330]:

adv


# ## value counts- Returns object containing counts of unique values.

# In[331]:

adv.news.value_counts()


# In[332]:

adv.label.value_counts()


# ## Aggregate statistics

# In[333]:

adv.describe()


# In[334]:

adv.groupby('label').describe()


# ## Removing Null values- Cleaning the data

# In[335]:

adv[adv.news.notnull()]
adv[adv.label.notnull()]


# In[336]:

adv=adv[pd.notnull(adv['news'])]
adv=adv[pd.notnull(adv['label'])]


# In[337]:

adv.isnull()


# ## Calculating the length of news

# In[338]:

adv['length']=adv['news'].map(lambda text: len(text))
adv.head(30)


# ## Plotting the graph according 

# In[339]:

adv.length.plot( bins=20, kind='hist')


# ## Plotting the histogram according to the length of both the labels

# In[340]:

adv.hist(column='length', by='label', bins=50)


# ## Data Preprocessing

# In[341]:

def tokenize(news):
    news2 = 'news -' + str(news)  # convert bytes into proper unicode
    return TextBlob(news).words


# In[342]:

adv.news.head().apply(tokenize)


# In[343]:

def lemmatize(news):
    news2 = 'news -' + str(news).lower()
    words = TextBlob(news).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

adv.news.head().apply(lemmatize)


# In[344]:

TextBlob("Strong Solar Storm, Tech Risks Today").tags 


# In[345]:

TextBlob("What's in that Iran bill that Obama doesn't like?").tags


# ## Data to Vectors- fitting and transforming using Count Vectorizer 

# In[346]:

bow_transformer=CountVectorizer(analyzer=lemmatize).fit(adv['news'])
len(bow_transformer.vocabulary_)


# In[347]:

news4=adv['news'][160]
news4


# In[348]:

bow4 = bow_transformer.transform([news4])
bow4


# In[349]:

bow4.shape


# #### //getting feature names

# In[350]:

bow_transformer.get_feature_names()[665]


# In[351]:

news_bow = bow_transformer.transform(adv['news'])

'sparsity: %.2f%%' % (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))


# In[352]:

'sparse matrix shape:', news_bow.shape


# In[353]:

'number of non-zeros:', news_bow.nnz


# ## Data to Vectors- fitting and transforming TFIDF- term frequency- inverse doc frequency and getting sparse matrix

# In[354]:

tfidf_transformer = TfidfTransformer().fit(news_bow)
tfidf4 = tfidf_transformer.transform(bow4)
tfidf4


# In[355]:

tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]


# In[356]:

news_tfidf = tfidf_transformer.transform(news_bow)
news_tfidf.shape


# ## Applying Multinomial on the whole training set and predicting accuracy

# In[357]:

get_ipython().magic("time spam_detector = MultinomialNB().fit(news_tfidf, adv['label'])")


# In[358]:

spam_detector=MultinomialNB().fit(news_tfidf, adv['label'])
spam_detector


# In[359]:

'predicted:', spam_detector.predict(tfidf4)[0]


# In[360]:

'expected:', adv.label[55]


# In[361]:

all_predictions = spam_detector.predict(news_tfidf)
all_predictions


# In[362]:

'accuracy', accuracy_score(adv['label'], all_predictions)


# In[363]:

'confusion matrix\n', confusion_matrix(adv['label'], all_predictions)


# In[364]:

'(row=expected, col=predicted)'


# In[365]:


plt.matshow(confusion_matrix(adv['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')


# In[366]:

print (classification_report(adv['label'], all_predictions))


# ## For Logistic Regression

# In[367]:

get_ipython().magic("time spam_detector = LogisticRegression().fit(news_tfidf, adv['label'])")


# In[368]:

spam_detector=LogisticRegression().fit(news_tfidf, adv['label'])
spam_detector


# In[369]:

print('predicted:', spam_detector.predict(tfidf4)[0])
print('expected:', adv.label[6])


# In[370]:

all_predictions = spam_detector.predict(news_tfidf)
all_predictions


# In[371]:

print('accuracy', accuracy_score(adv['label'], all_predictions))
print('confusion matrix\n', confusion_matrix(adv['label'], all_predictions))
print('(row=expected, col=predicted)')


# In[372]:


plt.matshow(confusion_matrix(adv['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')


# In[373]:

print (classification_report(adv['label'], all_predictions))


# ## Calculating how much data we are training and testing

# In[374]:

msg_train, msg_test, label_train, label_test = train_test_split(adv['news'], adv['label'], test_size=0.2)

len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


# #### Resulted in 5% of testing data and rest is the training data

# ## PIPELINE- to combine techniques
# 

# In[375]:

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='char')),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),# train on TF-IDF vectors w/ Naive Bayes classifie
])


# In[376]:

import _pickle as cPickle


# ## Cross Validation Scores for Logistic Regression

# In[377]:

scores = cross_val_score(pipeline,  # convert news into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[378]:

scores


# In[379]:

scores.mean(), scores.std()


# ## Cross Validation scores for Naive Bayes

# In[380]:

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='char')),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),# train on TF-IDF vectors w/ Naive Bayes classifie
])


# In[381]:

scores = cross_val_score(pipeline,  # convert news into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print(scores)


# In[382]:

scores.mean(), scores.std()


# In[383]:

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Data")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[384]:

get_ipython().magic('time plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)')


# In[385]:

from sklearn.grid_search import GridSearchCV


# ## GridSearch for SVM

# In[386]:

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer='char')),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1], 'classifier__kernel': ['linear']},
  {'classifier__C': [1], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)


# ## SVM time ad Scores

# In[387]:

get_ipython().magic('time svm_detector = grid_svm.fit(msg_train, label_train)')
svm_detector.grid_scores_


# In[388]:

print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
print(classification_report(label_test, svm_detector.predict(msg_test)))


# In[389]:

svm_detector.predict(["Donald Trump just trolled Rosie O'Donnell. Not good"])[0]


# In[390]:

svm_detector.predict(["Kushner family won't attend China investor pitch after criticism."])[0]


# In[391]:

svm_detector.predict(["US prosecuter told to push for more harsher punishments"])[0]


# In[392]:

clf=svm.SVC(kernel='linear', C=1.0,gamma=1)


# In[393]:

clf.fit(X_test_dtm,y_test)


# In[394]:

clf.score(X_test_dtm,y_test)


# In[395]:

predicted=clf.predict(X_test_dtm)


# In[396]:

predicted


# ## Count Vectorizer and TRAINING AND TESTING DATA

# In[397]:

vect=CountVectorizer()


# In[398]:

new_df1=adv[['news']]
new_df2=adv[['label']]


# In[399]:

train_data=new_df1.iloc[1:500,:]
test_data=new_df2.iloc[500:1,:]
train_label=new_df1.iloc[1:500,:]
test_label=new_df2.iloc[500:1,:]
train_vectors=cv.fit_transform(train_data)
test_vectors=cv.fit_transform(test_data)


# In[400]:

cv.get_feature_names()


# In[401]:

train_vectors.toarray()


# In[402]:

test_vectors.toarray()


# In[403]:

X=adv.news
y=adv.label


# In[404]:

print(X.shape)
print(y.shape)


# In[405]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,random_state=4)


# In[406]:

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[407]:

vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[408]:

X_train_dtm=vect.fit_transform(X_train)


# In[409]:

X_train_dtm


# In[410]:

X_test_dtm=vect.transform(X_test)
X_test_dtm


# ## Applying MACHINE LEARNING ALGORITHM ON TRAINING AND TESTING DATA

# # 1. KNN

# In[411]:

knn= KNeighborsClassifier(n_neighbors=8)


# In[412]:

knn.fit(X_train_dtm, y_train)


# In[413]:

y_pred_class=knn.predict(X_test_dtm)


# In[414]:

knn.score(X_test_dtm, y_test)


# In[415]:

get_ipython().magic('time knn.fit(X_train_dtm, y_train)')


# In[416]:

from sklearn import metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score
import sys
import scipy


# In[417]:

metrics.accuracy_score(y_test,y_pred_class)


# In[418]:

metrics.confusion_matrix(y_test, y_pred_class)


# In[419]:

print(metrics.classification_report(y_test, y_pred_class))


# In[420]:

scores = cross_val_score(KNeighborsClassifier(n_neighbors=15),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[421]:

scores


# In[422]:

scores.mean()


# In[423]:

scores.std()


# # 2.NAIVE BAYES

# In[424]:

nb=MultinomialNB()


# In[425]:

get_ipython().magic('time nb.fit(X_train_dtm, y_train)')


# In[426]:

nb.fit(X_train_dtm, y_train)


# In[427]:

y_pred_class=nb.predict(X_test_dtm)


# In[428]:

nb.score(X_train_dtm, y_train)


# In[429]:

metrics.confusion_matrix(y_test, y_pred_class)


# In[430]:

y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]
y_pred_prob


# In[431]:

metrics.accuracy_score(y_test, y_pred_class)


# In[432]:

print(metrics.classification_report(y_pred_class, y_test))


# In[433]:

scores = cross_val_score(MultinomialNB(),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[434]:

scores


# In[435]:

scores.mean()


# In[436]:

scores.std()


# # Logsitic Regression

# In[437]:

logreg=LogisticRegression()


# In[438]:

logreg.fit(X_train_dtm, y_train)


# In[439]:

y_pred_class=logreg.predict(X_test_dtm)


# In[440]:

logreg.score(X_test_dtm, y_test)



# In[441]:

get_ipython().magic('time logreg.fit(X_train_dtm, y_train)')


# In[442]:

metrics.accuracy_score(y_test,y_pred_class)



# In[443]:

metrics.confusion_matrix(y_test, y_pred_class)


# In[444]:

print(metrics.classification_report(y_test, y_pred_class))


# In[445]:

scores = cross_val_score(KNeighborsClassifier(n_neighbors=15),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[446]:

scores


# In[447]:

scores.mean()


# In[448]:

scores.std()


# In[449]:

names=['label','news','length']


# In[450]:

seed=7


# In[451]:

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC()))


# In[452]:

results=[]


# In[453]:

names=[]


# In[454]:

scoring='accuracy'


# In[455]:

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    scores = model_selection.cross_val_score(model, X_test_dtm, y_pred_class, cv=kfold, scoring=scoring)
    results.append(scores)
    names.append(name)
    msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
    print(msg)


# In[456]:

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[457]:

import matplotlib.pyplot as plt
 
# Data to plot
labels = 'Nave Bayes', 'SVM', 'K-NN', 'LG'
sizes = [80.14, 54.50, 59.77, 80.64]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


# In[460]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Nave Bayes', 'SVM', 'K-NN', 'LG')
y_pos = np.arange(len(objects))
performance = [7.02,6.13,4.01,19.1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time')
plt.title('Spam Detector Time')
 
plt.show()


# In[ ]:




# In[ ]:



