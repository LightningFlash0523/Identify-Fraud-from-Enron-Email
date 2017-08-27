
# coding: utf-8

# In[1]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split


# # Task 1: Select what features you'll use.

# In[2]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# ## The data

# In[3]:

df_data_dict = pd.DataFrame.from_dict(data_dict, orient='index', dtype = float)
df_data_dict.info()


# We have 146 observations with 21 variables in the dataset. There are also features with a lot of missing values.

# In[4]:

print 'There are {} POIs and {} non-POIs in the dataset.'.format(len(df_data_dict[df_data_dict.poi == 1]), len(df_data_dict[df_data_dict.poi == 0]))


# # Task 2: Remove outliers

# In[5]:

features_for_outliers = ["salary", "bonus"]
data = featureFormat(data_dict, features_for_outliers)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[6]:

"""
There is an obvious one outlier based on the plot above. 
And upon checking the enron61702insiderpay.pdf file, I found out that this value refers to the row 'TOTAL'.
This is not a valid data point so it must be removed from our dataset.
"""
data_dict.pop("TOTAL", 0 ) 


# In[7]:

"""
After removing the 'TOTAL' using data_dict.pop("TOTAL", 0 ) , I have now a new plot.
"""
features_for_outliers = ["salary", "bonus"]
data = featureFormat(data_dict, features_for_outliers)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[8]:

"""
It is much better now but there are still 4 points that could be also outliers. Those must be investigated. Two people made 
bonuses of at least 5 million dollars, and a salary of over 1 million dollars. These could be potential outliers.
"""
df_data_dict.sort_values('salary', ascending = False).head(2)


# In[9]:

"""
They are the big bosses of Enron, and definitely poi's, so the are valid data points. It is best to leave them in as part 
of the dataset.
"""
pass


# # Task 3: Create new feature(s)

# In[10]:

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    try:
        fraction = poi_messages*1.0 / all_messages
    except:
        fraction = 0
        
    return fraction

for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi


# In[11]:

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# splitting my dataset into training and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# # Task 4: Try a variety of classifiers

# In[12]:

def clf_performance(classifier, X_train, Y_train, X_test, Y_test):
    # training my training set
    t0 = time()
    classifier.fit(X_train , Y_train)
    print "training time:", round(time()-t0, 3), "s"
    
    # predicting from my feature_test
    t0 = time()
    prediction = classifier.predict(X_test)
    print "prediction time:", round(time()-t0, 3), "s"   
    print 'Accuracy: ', round(accuracy_score(prediction, Y_test),3)    
    print 'precision = ', round(precision_score(Y_test,prediction),3)
    print 'recall = ', round(recall_score(Y_test,prediction),3)


# In[13]:

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf_performance(clf, features_train, labels_train, features_test, labels_test)
"""
training time: 0.001 s
prediction time: 0.001 s
Accuracy:  0.879
precision =  0.0
recall =  0.0
"""
pass


# In[14]:

from sklearn.svm import SVC
clf = SVC()
clf_performance(clf, features_train, labels_train, features_test, labels_test)
"""
training time: 0.002 s
prediction time: 0.001 s
Accuracy:  0.909
precision =  0.0
recall =  0.0
"""
pass


# In[15]:

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf_performance(clf, features_train, labels_train, features_test, labels_test)
"""
training time: 0.004 s
prediction time: 0.0 s
Accuracy:  0.818
precision =  0.2
recall =  0.2
"""
pass


# # Updating features

# In[16]:

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)

importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Importance Ranking: '
for i in range(16):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


# In[17]:

#features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
#                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
#                 'deferred_income', 'total_stock_value']
features_list = ["poi", "shared_receipt_with_poi", "fraction_from_poi", "fraction_to_poi"]

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

clf = DecisionTreeClassifier()
clf_performance(clf, features_train, labels_train, features_test, labels_test)


# # Task 5: Tune your classifier

# In[18]:

param_grid = {
          'criterion': ['gini', 'entropy'],
          'min_samples_split': [2, 5, 10, 15, 20, 25],
          }
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf_performance(clf, features_train, labels_train, features_test, labels_test)


# In[25]:

# checking out the final model without the "new features"
features_list = ["poi", "salary", "bonus", 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf_performance(clf, features_train, labels_train, features_test, labels_test)


# In[27]:

# checking out the final model without the "new features"
features_list = ["poi", "salary", "bonus", 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf_performance(clf, features_train, labels_train, features_test, labels_test)


# In[28]:

#Checking out the final model with the "new features"
features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi", 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf_performance(clf, features_train, labels_train, features_test, labels_test)


# In[36]:

### This one gave the best result:
features_list = ["poi", "shared_receipt_with_poi", "fraction_from_poi", "fraction_to_poi"]

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf_performance(clf, features_train, labels_train, features_test, labels_test)
print clf.best_estimator_


# # Task 6: Dump your classifier, dataset, and features_list

# In[20]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



