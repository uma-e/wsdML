from sys import argv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

#quick word sense disambiguation using ml models

# Usage: python mlwsd.py line-train.txt line-test.txt [Optional: model_choice] my-model.txt > my-line-answers.txt

#Word sense disambiguation (WSD) is the task of determining the correct sense of a word within a given context. 
#Words can have multiple meanings depending on the context in which they are used, and WSD aims to automatically 
#identify the correct sense of a word in a particular instance by analyzing its surrounding words or context. 
#This task is fundamental in nlp  and is crucial for various applications such as 
#machine translation, information retrieval, and semantic analysis.
#In this implementation, WSD is performed using machine learning models. 

#Multinomial Naive Bayes:
#uses Bayes' theorem with the assumption of independence between features to 
#calculate the probability of a class given a set of features
#It assigns weights to each feature based on their occurrences in each class.
#Given a new instance, it calculates the likelihood of each class and selects the class with 
#the highest likelihood as the prediction.

#Support Vector Machines (SVM):
#Makes a hyperplane in a high-dimensional space that separates instances of 
#different classes with the largest margin. It transforms the input data into a higher-dimensional 
#space using a kernel function to make the data linearly separable if it's not in the original space 
#It then finds the hyperplane that maximizes the margin between classes

#Gradient Boosted Trees:
#Creates an ensemble of decision trees sequentially where each tree corrects the errors made by the 
#previous ones. It fits a simple model to the data and then adds more models
#each one focusing on the instances that the previous models predicted incorrectly
#It combines the predictions of all trees to make the final prediction

#Algorithm performs wsd by using machine learning models from scikit-learn 
#It begins by checking if a model choice is provided in the user input
#It selects the appropriate model based on the choice with the default being multimodal naive Bayes
#The training data is read, preprocessed, and then the senses are extracted from train
#The contexts are further preprocessed, and features are extracted using the CountVectorizer from scikit-learn.
#the countVectorizer allows the models to work with the text data a little better by making a matrix of tokens
#The selected model is trained on the extracted features using the fit function
#The algorithm reads the test data from the file preprocesses the test contexts and transforms again with countVectorizer
#the algorithm makes predictions using the trained model and outputs the predictions

train = str(argv[1])
test = str(argv[2])

#optional third arg to specify model
if len(argv) > 3:
    model_choice = str(argv[3])
else:
    model_choice = None

#determine the model to use
#default is NB
if model_choice and model_choice.lower() == 'svm':
    model = SVC()  
elif model_choice and model_choice.lower() == 'gradientboost':
    model = GradientBoostingClassifier()
else:
    model = MultinomialNB() 

#preprocess train file by removing unnecessary tags and splitting it into relevant sections
with open(train, 'r') as file:
    trainingSet = file.read()
trainingSet = re.sub(r'<[/]?context>\s|</instance>\s', '', trainingSet)
trainingSet = trainingSet.splitlines()
trainingSet = trainingSet[2:-2]

#extract the senses from the training data and count their occurrences
senses = []
for i in range(1, len(trainingSet), 3):
    x = trainingSet[i]
    senseID = re.findall('senseid="([^"]*)"', x)
    senses.append(senseID[0])

#preprocess the training contexts by replacing specific tags in a standardized format
tContext = [re.sub(r"\<head\>lines\<\/head\>", "<head>line</head>", x) for x in trainingSet[2::3]]

#convert contexts into numerical feature vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(tContext)

#train the ML model using the training data
model.fit(X_train, senses)

#load testing data and preprocess it
with open(test, 'r') as testfile:
    testList = testfile.read()
    testList = re.sub(r'<[/]?context>\s|</instance>\s', '', testList)
    testList = testList.splitlines()
    testList = testList[2:len(testList) - 2]

#preprocess test contexts and turn into numerical feature vectors
testCon = [re.sub(r"\<head\>lines\<\/head\>", "<head>line</head>", raw.lower().replace('<s>', '').replace('</s>', '').replace('<@>', '').replace('<p>', '').replace('</p>', '')) for raw in testList[1::2]]
X_test = vectorizer.transform(testCon)


#use trained model to predict the test data
predictions = model.predict(X_test)

#outputting predictions
for instance_id, prediction in enumerate(predictions, start=1):
    instance_id_str = re.findall('id="([^"]*)"', testList[2*(instance_id-1)])
    print(f"<answer instance=\"{instance_id_str[0]}\" senseid=\"{prediction}\"/>")
