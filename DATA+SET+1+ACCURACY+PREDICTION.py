
# coding: utf-8

# ## Importng the packages and modules required in the project

# In[1]:

import pandas as pd
import numpy as np
import csv
import sys
import scipy
from sklearn import naive_bayes
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.learning_curve import learning_curve
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger


# ## Reading data - csv file using pandas

# In[2]:

adv=pd.read_csv('fake_22.csv')


# ## How the data frame looks like

# In[3]:

adv.shape


# ## First 10 values of data frame

# In[4]:

adv.head(10)


# ## value counts- Returns object containing counts of unique values.

# In[5]:

adv.title.value_counts()


# In[6]:

adv.author.value_counts()


# In[7]:

adv.language.value_counts()


# In[8]:

adv.country.value_counts()


# In[9]:

adv.site_url.value_counts()


# In[10]:

adv.domain_rank.value_counts()


# In[11]:

adv.spam_score.value_counts()


# In[12]:

adv.type.value_counts()


# ## Aggregate statistics

# In[149]:

adv.describe()


# In[150]:

adv.groupby('type').describe()


# In[13]:

new_data=adv[['author','title','language','country','site_url','domain_rank','spam_score','type']]


# In[14]:

new_data


# ## Removing Null values- Cleaning the data

# In[15]:

adv[adv.author.notnull()]
adv[adv.title.notnull()]
adv[adv.language.notnull()]
adv[adv.site_url.notnull()]
adv[adv.country.notnull()]
adv[adv.domain_rank.notnull()]
adv[adv.spam_score.notnull()]
adv[adv.type.notnull()]


# In[16]:

adv=adv[pd.notnull(adv['author'])]
adv=adv[pd.notnull(adv['title'])]
adv=adv[pd.notnull(adv['language'])]
adv=adv[pd.notnull(adv['country'])]
adv=adv[pd.notnull(adv['site_url'])]
adv=adv[pd.notnull(adv['domain_rank'])]
adv=adv[pd.notnull(adv['spam_score'])]
adv=adv[pd.notnull(adv['type'])]


# In[17]:

adv.isnull()


# In[18]:

len(adv)


# In[19]:

import matplotlib.pyplot as plt


# In[20]:

get_ipython().magic('matplotlib inline')


# ## Length Calculation

# In[21]:

adv['length']=adv['title'].map(lambda text: len(text))
adv.head()


# ## Plotting the graph according 

# In[22]:

adv.length.plot(bins=20, kind='hist')


# In[23]:

adv.length.describe()


# ## Length comaprisons for different types

# In[24]:

print(adv.hist(column='length', by='type', bins=30))


# ## Processing features

# In[25]:

def tokenize(title):
    title2 = 'title -' + str(title)  # convert bytes into proper unicode
    return TextBlob(title).words


# In[26]:

adv.title.head().apply(tokenize)


# In[27]:

def lemmatize(title):
    title2 = 'title -' + str(title).lower()
    words = TextBlob(title).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

adv.title.head().apply(lemmatize)


# In[28]:

TextBlob("BREAKING: Weiner Cooperating With FBI On Hilla").tags


# ## Data to Vectors- fitting and transforming using Count Vectorizer 

# In[29]:

bow_transformer = CountVectorizer(analyzer=lemmatize).fit(adv['title'])
len(bow_transformer.vocabulary_)


# In[30]:

title4=adv['title'][20]


# In[31]:

title4


# In[32]:

bow4 = bow_transformer.transform([title4])
bow4


# In[33]:

bow_transformer.get_feature_names()[456]


# ## SParsity- sparse matrix

# In[34]:

title_bow = bow_transformer.transform(adv['title'])

'sparsity: %.2f%%' % (100.0 * title_bow.nnz / (title_bow.shape[0] * title_bow.shape[1]))


# In[35]:

'sparse matrix shape:', title_bow.shape


# In[36]:

'number of non-zeros:', title_bow.nnz


# ## Data to Vectors- fitting and transforming TFIDF- term frequency- inverse doc frequency and getting sparse matrix

# In[37]:

tfidf_transformer = TfidfTransformer().fit(title_bow)
tfidf4 = tfidf_transformer.transform(bow4)
tfidf4


# In[38]:

tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]


# In[39]:

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[40]:

title_tfidf = tfidf_transformer.transform(title_bow)
title_tfidf.shape


# ## Applying Multinomial on the whole training set and predicting accuracy

# In[41]:

get_ipython().magic("time spam_detector = MultinomialNB().fit(title_tfidf, adv['type'])")


# In[42]:

spam_detector=MultinomialNB().fit(title_tfidf, adv['type'])


# In[43]:

spam_detector


# In[44]:

'predicted:', spam_detector.predict(tfidf4)[0]


# In[45]:

'expected:', adv.type[8]


# In[46]:

'expected:', adv.type[500]


# In[47]:

all_predictions = spam_detector.predict(title_tfidf)
all_predictions


# In[48]:

'accuracy', accuracy_score(adv['type'], all_predictions)


# In[49]:

print('confusion matrix\n', confusion_matrix(adv['type'], all_predictions))


# In[50]:

'(row=expected, col=predicted)'


# In[51]:

plt.matshow(confusion_matrix(adv['type'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')


# In[52]:

print (classification_report(adv['type'], all_predictions))


# ## Calculating how much data we are training and testing

# In[53]:

msg_train, msg_test, label_train, label_test = train_test_split(adv['title'], adv['type'], test_size=0.2)

len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


# ## PIPELINE- to combine techniques
# 

# In[54]:

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='char')),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[55]:

import _pickle as cPickle


# In[56]:

scores = cross_val_score(pipeline,  # steps to convert raw emails into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# ## Cross validation scores

# In[57]:

scores


# In[58]:

scores.mean(), scores.std()


# In[59]:

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


# In[60]:

get_ipython().magic('time plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)')


# ## SVM

# In[61]:

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


# In[62]:

get_ipython().magic('time svm_detector = grid_svm.fit(msg_train, label_train)')
svm_detector.grid_scores_


# In[63]:

print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
print(classification_report(label_test, svm_detector.predict(msg_test)))


# In[64]:

svm_detector.predict(["Trump and Brexit: Directed History Proceeds"])[0]


# In[65]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split


# ## Count Vectorizer and TRAINING AND TESTING DATA

# In[66]:

cv=CountVectorizer()


# In[67]:

new_df1=adv[['title','author','language','country','domain_rank','site_url','spam_score']]


# In[68]:

new_df2=adv[['type']]


# In[69]:

train_data=new_df1.iloc[1:3000,:]


# In[70]:

test_data=new_df1.iloc[3000:1,:]


# In[71]:

train_label=new_df2.iloc[1:3000,:]


# In[72]:

test_label=new_df1.iloc[3000:1,]


# In[73]:

train_vectors=cv.fit_transform(train_data)


# In[74]:

test_vectors=cv.fit_transform(test_data)


# In[75]:

cv.get_feature_names()


# In[76]:

adv.groupby('type').describe()


# In[77]:

cv


# In[78]:

adv['type_num']=adv.type.map({'fake':0,'conspiracy':1,'satire':2, 'bias':3,'hate':4})


# In[79]:

adv


# In[80]:

X=adv.title
y=adv.type_num


# In[81]:

print(X.shape)
print(y.shape)


# In[82]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,random_state=4)


# In[83]:

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[84]:

vect = CountVectorizer()


# In[85]:

from textblob import TextBlob


# In[86]:

def split_into_tokens(title):
    message = unicode(title, 'utf8')  # convert bytes into proper unicode
    return TextBlob(title).words


# In[87]:

TextBlob('title').tags


# In[88]:

vect.fit(X_train)
X_train_dtm = vect.transform(X_train)


# In[89]:

X_train_dtm=vect.fit_transform(X_train)


# In[90]:

X_train_dtm


# In[91]:

X_test_dtm=vect.transform(X_test)
X_test_dtm


# ## K Nearest Neighbors

# In[92]:

from sklearn.neighbors import KNeighborsClassifier


# In[93]:

knn=KNeighborsClassifier(n_neighbors=15)


# In[94]:

knn.fit(X_train_dtm, y_train)


# In[95]:

y_pred_class=knn.predict(X_test_dtm)


# In[96]:

y_pred_class


# In[100]:

print("Test set predictions:\n {}".format(y_pred_class))


# In[101]:

knn.score(X_test_dtm, y_test)


# In[102]:

get_ipython().magic('time knn.fit(X_train_dtm, y_train)')


# In[106]:

from sklearn import metrics


# In[107]:

metrics.accuracy_score(y_test,y_pred_class)


# In[108]:

metrics.confusion_matrix(y_test, y_pred_class)


# In[109]:

print(metrics.classification_report(y_test, y_pred_class))


# In[110]:

scores = cross_val_score(KNeighborsClassifier(n_neighbors=15),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[111]:

scores


# In[112]:

scores.mean()


# ## Naive Bayes

# In[113]:

from sklearn.naive_bayes import MultinomialNB


# In[114]:

nb=MultinomialNB()


# In[115]:

get_ipython().magic('time nb.fit(X_train_dtm, y_train)')


# In[116]:

y_pred_class=nb.predict(X_test_dtm)


# In[117]:

from sklearn import metrics


# In[118]:

metrics.confusion_matrix(y_test, y_pred_class)


# In[119]:

X_test[4600]


# In[120]:

nb.predict_proba(X_test_dtm)[:,1]


# In[121]:

y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]


# In[122]:

y_pred_prob


# In[123]:

nb.score(X_train_dtm, y_train)


# In[124]:

metrics.accuracy_score(y_test, y_pred_class)


# In[125]:

print(metrics.classification_report(y_pred_class, y_test))


# In[126]:

scores = cross_val_score(MultinomialNB(),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[127]:

scores


# In[128]:

scores.mean()


# ## Logistic Regression

# In[129]:

from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()


# In[130]:

get_ipython().magic('time logreg.fit(X_train_dtm, y_train)')


# In[131]:

logreg.score(X_train_dtm, y_train)


# In[132]:

y.mean()


# In[133]:

y_pred_class=logreg.predict(X_test_dtm)


# In[134]:

y_pred_prob=logreg.predict_proba(X_test_dtm)
y_pred_prob


# In[135]:

metrics.accuracy_score(y_test, y_pred_class)


# In[136]:

print(metrics.classification_report(y_test, y_pred_class))


# In[137]:

scores = cross_val_score(LogisticRegression(),  # steps to convert raw emails into models
                         X_train_dtm,  # training data
                         y_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )


# In[138]:

scores


# In[139]:

scores.mean()


# ## RESULTS AND ANALYSIS

# In[140]:

names=['author','title','language', 'country', 'site_url', 'domain_rank', 'spam_score', 'type']


# In[141]:

seed=7


# In[142]:

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC()))


# In[143]:

results=[]


# In[144]:

names=[]


# In[145]:

scoring='accuracy'


# In[146]:

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    scores = model_selection.cross_val_score(model, X_test_dtm, y_pred_class, cv=kfold, scoring=scoring)
    results.append(scores)
    names.append(name)
    msg = "%s: %f (%f)" % (name, scores.mean(), scores.std())
    print(msg)


# In[147]:

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[151]:

import matplotlib.pyplot as plt
 
# Data to plot
labels = 'Nave Bayes', 'SVM', 'K-NN', 'LG'
sizes = [87, 86, 85, 85.33]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[152]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Nave Bayes', 'SVM', 'K-NN', 'LG')
y_pos = np.arange(len(objects))
performance = [4.01,6.88,5.5,65.2]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time')
plt.title('Spam Detector Time')
 
plt.show()


# In[ ]:



