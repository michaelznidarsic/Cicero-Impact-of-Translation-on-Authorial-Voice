# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 01:54:39 2019

@author: mznid
"""

##### CICERO LETTERS
import os
import re
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

os.getcwd()
path = "C:\\Users\\mznid\\Downloads\\CiceroLettersCorpus"
os.chdir(path)


with open('1.txt', encoding='utf-8', errors='ignore') as f:
   lines = f.readlines()


storedlines = '' 
meta = ''
letter = [] 
metadata = []
descriptions = [] 
copy = False 
standby = False
descriptioncounter = 0
footnotes = False
for line in lines: 
    line = line.rstrip('\n')
    #line = line.rstrip() #remove the white spaces etc at the end of the line 
    if re.search(u'^[A-Z]+ [(][A-Z] [A-Z]+[,] \d+', line) or re.search(u'^[A-Z]+ [(][A-Z] [A-Z]+ [A-Z]+[,] \d+', line): 
        letter.append(line)
        if copy:
            descriptions.append(storedlines)
            copy = False
        elif footnotes:
            descriptions.append(storedlines)
            footnotes = False
        standby = True
        storedlines = '' 
    elif standby:
        if re.search(u'^TO ', line):
            standby = False
            descriptioncounter += 2
            meta = meta + ' ' + line
        elif re.search(u'^FROM ', line):
            standby = False
            letter.pop()
    elif descriptioncounter > 0:
        descriptioncounter -= 1
        meta = meta + ' ' + line
        if descriptioncounter == 0:
            metadata.append(meta)
            meta = ''
            copy = True
    elif re.search(u'\[Footnote 734', line):
        copy = False
        descriptions.append(storedlines)
        storedlines = ''
        break
    elif copy: 
        if re.search(u'^\[Footnote ', line):
            copy = False
            footnotes = True
        else:
            storedlines = storedlines + ' ' + line


print(letter)
print(metadata)
print(descriptions[0])

len(letter)
len(metadata)
len(descriptions)


data = {'Letter' : letter, 'Metadata' : metadata}
newmetadf = pd.DataFrame(data)
print(newmetadf)


import pandas as pd

data = {'Letter' : letter, 'Metadata' : metadata}
metadf = pd.DataFrame(data)
for each in letter:
    print(len(each))

print(lines)

newmetadf.to_csv('anothermetadf.csv')
print(metadf)



quintusindex = []
atticusindex = []
for index, each in enumerate(metadata):
    if 'TO HIS BROTHER QUINTUS' in each:
        quintusindex.append(index)
    if 'TO ATTICUS' in each:
        atticusindex.append(index)


len(quintusindex)
len(atticusindex)

rawdata = {'ID': letter, 'meta': metadata,'text': descriptions}
rawdf = pd.DataFrame(rawdata)

quintus = rawdf.iloc[quintusindex,:]
atticus = rawdf.iloc[atticusindex,:]
print(atticus.iloc[73,2])

newcol = []
for each in quintus.iloc[:,2]:
    newcol.append('Q')
quintus.insert(0, "label", newcol, True) 

newcol = []
for each in atticus.iloc[:,2]:
    newcol.append('A')
atticus.insert(0, "label", newcol, True) 

labelled = pd.concat([quintus.iloc[:,[0,3]], atticus.iloc[:,[0,3]]])


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


##### EXCLUDES GREEK CHARACTERS AND STUFF, 4-20 CHARACTER TOKEN LENGTH
invective = CountVectorizer(input = 'content', token_pattern = r'\b[^\d\W]{4,20}\b') #token_pattern = r'\b[^\d\W]{4,20}\b'

result = invective.fit_transform(labelled.iloc[:,1])

dtmdf = pd.DataFrame(result.A, columns=invective.get_feature_names())


tfidfdf = TfidfTransformer(use_idf=False).fit(dtmdf)
newdf = tfidfdf.transform(dtmdf)
newdf = pd.DataFrame(newdf.A, columns=invective.get_feature_names())


##############TFIDF TOGGLE############################################################################
dtmdf = newdf
######################################################################################################


labels = labelled.iloc[:,0]



###################
print(list(dtmdf.columns))

print(dtmdf.columns[7666])
removegreek = dtmdf.iloc[:,1:7667]

############## DON'T REMOVE GREEK:
removegreek = dtmdf
##########################


stopwordnames = ['quintus', 'atticus']

removegreek.drop(columns=stopwordnames)


# 68BC - 52BC


newquintusindex = list(range(0, 27))
newatticusindex = list(range(27, 119))

import random 

balancer = random.sample(newatticusindex,27) 



quintusclean = removegreek.iloc[0:27,:]
atticusclean = removegreek.iloc[balancer,:]

quintustestindex = random.sample(list(range(0,27)),8)
atticustestindex = random.sample(list(range(0,27)),8)

quintustrainindex = list( set(range(0,27)) - set(quintustestindex) )
atticustrainindex = list( set(range(0,27)) - set(atticustestindex) )

test = pd.concat([quintusclean.iloc[quintustestindex,:],atticusclean.iloc[atticustestindex,:]])
train = pd.concat( [ quintusclean.iloc[quintustrainindex,:] , atticusclean.iloc[atticustrainindex,:] ] )



import numpy as np


labeltest = []

for each in quintustestindex:
    labeltest.append('Q')
for each in atticustestindex:
    labeltest.append('A')


labeltrain = []

for each in quintustrainindex:
    labeltrain.append('Q')

for each in atticustrainindex:
    labeltrain.append('A')

########## BEGIN DECISION TREE ANALYSIS


from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn import tree

clf = DecisionTreeClassifier(min_samples_split = 3, min_samples_leaf = 2, max_depth = 8)


clf = clf.fit(train,labeltrain)


testpred = clf.predict(test)



print("Accuracy:", metrics.accuracy_score(labeltest, list(testpred)))

treecorrect = list(testpred == labeltest)

sum(treecorrect)/len(treecorrect)


trainpred = clf.predict(train)
treecorrect = list(trainpred == labeltrain)

sum(treecorrect)/len(treecorrect)


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names = removegreek.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('CICEROTREE.png')
Image(graph.create_png())







actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(testpred, dtype= "category")

confusionmatrix = pd.crosstab(actual, pred)

type(confusionmatrix)



sb.heatmap(confusionmatrix, annot=True, cmap="Blues")
plt.xlabel('Prediction')
plt.ylabel('Actual')





unused = removegreek.iloc[~removegreek.index.isin(newquintusindex + balancer)]

unusedpredict = clf.predict(unused)

sum(list(unusedpredict) == 'A' )/ len(list(unusedpredict))


correctcounter = 0
for each in list(unusedpredict):
    if each == 'A':
        correctcounter += 1
print( correctcounter / len(list(unusedpredict)) )





######## BEGIN NAIVE BAYES ANALYSIS


from sklearn.naive_bayes import MultinomialNB

MyModelNB = MultinomialNB()

MyModelNB.fit(train, labeltrain)

prediction = MyModelNB.predict(test)




np.round(MyModelNB.predict_proba(test),2)


correct = list(prediction == labeltest)


sum(correct)/len(correct)


trainprediction = MyModelNB.predict(train)
traincorrect = list(trainprediction == labeltrain)
sum(traincorrect)/len(traincorrect)



actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(prediction, dtype= "category")

confusionmatrix = pd.crosstab(actual, pred)

type(confusionmatrix)



sb.heatmap(confusionmatrix, annot=True, cmap="Blues")
plt.xlabel('Prediction')
plt.ylabel('Actual')




unused = removegreek.iloc[~removegreek.index.isin(newquintusindex + balancer)]

unusedpredict = MyModelNB.predict(unused)

sum(list(unusedpredict) == 'A' )/ len(list(unusedpredict))

correctcounter = 0
for each in list(unusedpredict):
    if each == 'A':
        correctcounter += 1
print( correctcounter / len(list(unusedpredict)) )



import pickle

filename = 'ENGLISHNBmodel.sav'
#pickle.dump(MyModelNB, open(filename, 'wb'))

 