import re
import pandas as pd
from sys import argv

# Usage: python scorer.py my-line-answers.txt line-key.txt
# this program compares a list of model predictions with a list of answer keys to evaluate the accuracy of the model to 
# create a confusion matrix and find the overall accuracy

# This program serves as a scorer for evaluating the accuracy of a wsd model.
# Given a list of model predictions and a corresponding list of answer keys, it compares each prediction with the
# corresponding answer key and calculates the accuracy of the model. The program also generates a confusion matrix
# to provide further insights into the performance of the model.

# Algorithm:
# read model predictions and answer keys from input files
# iterate through each prediction and compare it with the corresponding answer key
# append the actual and predicted senses to lists
# calculate the accuracy of the model based on the number of correct predictions
# create series from lists to construct a confusion matrix
# generate the confusion matrix using pandas




guesses, key = str(argv[1]), str(argv[2])
correct, incorrect = 0, 0
myList, keyList, true, pred = [], [], [], []

with open(guesses, "r", encoding="utf-16") as file1:
    myList = [line.strip() for line in file1]

with open(key, "r") as file2:
    keyList = [line.strip() for line in file2]

#compare predictions with answer keys
for i in range(len(keyList)):
    try:
        if myList[i] == keyList[i]:  
            correct += 1  
            #append senses based on regex search
            if re.search(r'phone', keyList[i]):
                true.append('phone'); pred.append('phone')
            else:
                true.append('product'); pred.append('product')
        else:
            #append senses and set opposite prediction
            if re.search(r'phone', keyList[i]):
                true.append('phone'); pred.append('product')
            else:
                true.append('product'); pred.append('phone')
            incorrect += 1  
    except:
        print("Error: Lists have different sizes.")

#calculate accuracy
total = correct + incorrect
accuracy = (correct / total) * 100

print("Accuracy: {:.2f}%".format(accuracy))

trues = pd.Series(true, name='True')
preds = pd.Series(pred, name='Predicted')
matrix = pd.crosstab(preds, trues)
print("\nconfusion matrix:")
print(matrix)
