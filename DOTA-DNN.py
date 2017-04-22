#Dataset can be optained from UCI Machine Learning Repo.
#Import the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from sklearn import preprocessing

tf.logging.set_verbosity(tf.logging.INFO)
from sklearn import decomposition

#Model Features
a = 5               #Number of Neurons in Layer 1
b = 5				#Number of Neurons in Layer 2
num_classes = 2		#Number of Training CLasses
num_steps = 15000	#Trainsteps or iteration

#Define the optimizer for the network
optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)


#Read the data file
train = pd.read_csv('./data/dota/dota2Train.csv', header=None)
print ("Shape of Train data:",train.shape)

test = pd.read_csv('./data/dota/dota2Test.csv', header=None)
print ("rain shape:",train.shape)
print ("test shape:",test.shape)


#Seperate the feature and target variables
coloumns = (train.columns)
features = train.ix[:3,1:]
features = features.columns
coloumns = list(coloumns)
features = list(features)

#Create Training and Test Set
training_set = train
test_set = test
print(training_set.shape)
print(test_set.shape)


#Since No Labels we provided with the data set we need to Assign column/features names as we should not keet them numeric
lb = preprocessing.LabelEncoder()
f1 = lb.fit_transform(coloumns)
f2 = lb.transform(features)
f1 = f1.astype('str')
f2 = f2.astype('str')
f1 = list(f1)
f2= list(f2)

f1 = ["a" + f1 for f1 in f1]
f2 = ["a" + f2 for f2 in f2]

#Parameters to be used in Training
COLUMNS = f1
FEATURES = f2
LABEL = 'a0'
training_set.columns = f1
test_set.columns = f1

#Target label contains -1 as class we will convert it to 0 just to be safe
training_set.a0 = training_set.a0.replace(to_replace=[-1], value=[0])
test_set.a0 = test_set.a0.replace(to_replace=[-1], value=[0])

#creating input funtion to feed to the DNN
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

#defining the classifier
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,hidden_units=[a,b], model_dir='./model/',n_classes=num_classes,
                                           optimizer=optimizer)

#Lets fit the classifier to our data
classifier.fit(input_fn=lambda: input_fn(training_set),steps=num_steps)

#Model Evaluation
ev = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=500)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))