# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:50:08 2019

@author: mznid
"""
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn import tree
import sklearn.svm as SVM
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


from cltk.stem.lemma import LemmaReplacer
from cltk.corpus.utils.importer import CorpusImporter
corpus_importer = CorpusImporter('latin')
corpus_importer.import_corpus('latin_models_cltk')

path = "C:\\Users\\mznid\\Downloads\\CiceroLettersCorpus\\New"
os.chdir(path)

with open('QUINTUS.txt', encoding='utf-8', errors='ignore') as f:
   # Read the file contents and generate a list with each line
   QUINTUSTEXT = f.readlines()

with open('ATTICUS.txt', encoding='utf-8', errors='ignore') as f:
   # Read the file contents and generate a list with each line
   ATTICUSTEXT = f.readlines()




letter = []
contents = []
storedlines = ''
copy = False
standby = False
meta = []


for line in QUINTUSTEXT: 
    line = line.rstrip('\n')
    #line = line.rstrip() #remove the white spaces etc at the end of the line 
    if re.search(u'^[A-Z]+\. Scr\.', line): 
        letter.append(line)
        if copy:
            contents.append(storedlines)
            copy = False
        storedlines = ''
        standby = True
    elif standby:
        meta.append(line)
        standby = False
        copy = True
    elif copy:
        storedlines = storedlines + ' ' + line
contents.append(storedlines)
storedlines = ''

len(contents)
len(letter)

data = {'Letter': letter, 'Contents': contents}
QUINTUSDF = pd.DataFrame(data)
QUINTUSDF.shape




letter = []
contents = []
storedlines = ''
copy = False
meta = []
standbycounter = 0

for line in ATTICUSTEXT: 
    line = line.rstrip('\n')  
    if re.search(u'^I\.[A-Z]+ ', line):
        letter.append(line)
        if copy:
            contents.append(storedlines)
            copy = False
        storedlines = ''
        standbycounter = 2
    elif re.search(u'^\[[A-Z]+\] Scr\.', line):
        letter.append(line)
        if copy:
            contents.append(storedlines)
            copy = False
        storedlines = ''
        standbycounter = 1
    elif standbycounter > 0:
        meta.append(line)
        standby = False
        standbycounter -= 1
        if standbycounter == 0:
            copy = True
    elif copy:
        storedlines = storedlines + ' ' + line        
contents.append(storedlines)
storedlines = ''



len(contents)
len(letter) 

data = {'Letter': letter, 'Contents': contents}
ATTICUSDF = pd.DataFrame(data)
ATTICUSDF.shape         

print(ATTICUSDF)
labels = []
for index, blah in QUINTUSDF.iterrows():
    labels.append('Q')
for index, blah in ATTICUSDF.iterrows():
    labels.append('A')
    
labelled = pd.concat([QUINTUSDF, ATTICUSDF])
labelled.insert(0, 'label', labels)


############# PREPARE LEMMATIZING

lemmatizer = LemmaReplacer('latin')
# pass as tokenizer = lemmatizer.lemmatize

analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (lemmatizer.lemmatize(w) for w in analyzer(doc))
# pass as analyzer = stemmed_words

#############


vectorvaughn = CountVectorizer(input = 'content', token_pattern = r'\b[^\d\W]{4,20}\b', tokenizer = lemmatizer.lemmatize)

result = vectorvaughn.fit_transform(labelled.iloc[:,2])

dtmdf = pd.DataFrame(result.A, columns=vectorvaughn.get_feature_names())



######################## WORDCLOUD VIZ

seperator = ', '
wordcloud = WordCloud().generate(str(seperator.join(contents)))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



########################## NORMALIZING TOGGLE

tfidfdf = TfidfTransformer(use_idf=False).fit(dtmdf)
newdf = tfidfdf.transform(dtmdf)
newdf = pd.DataFrame(newdf.A, columns=vectorvaughn.get_feature_names())

dtmdf = newdf


#########################







##### SPLIT TEST/TRAIN

newquintusindex = list(range(0, 26))
newatticusindex = list(range(26, 115))


balancer = random.sample(newatticusindex,26)

quintusclean = dtmdf.iloc[0:26,:]
atticusclean = dtmdf.iloc[balancer,:]

quintustestindex = random.sample(list(range(0,26)),8)
atticustestindex = random.sample(list(range(0,26)),8)

quintustrainindex = list( set(range(0,26)) - set(quintustestindex) )
atticustrainindex = list( set(range(0,26)) - set(atticustestindex) ) 


test = pd.concat([quintusclean.iloc[quintustestindex,:],atticusclean.iloc[atticustestindex,:]])
train = pd.concat( [ quintusclean.iloc[quintustrainindex,:] , atticusclean.iloc[atticustrainindex,:] ] )



####### LABEL CREATION

labeltest = []

for each in quintustestindex:
    labeltest.append("Q")
for each in atticustestindex:
    labeltest.append("A")
    
labeltrain = []

for each in quintustrainindex:
    labeltrain.append("Q")
for each in atticustrainindex:
    labeltrain.append("A")









#### NB MODEL

MyModelNB = MultinomialNB()

MyModelNB.fit(train, labeltrain)

prediction = MyModelNB.predict(test)



np.round(MyModelNB.predict_proba(test),2)
correct = list(prediction == labeltest)
sum(correct)/len(correct)


actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(prediction, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
type(confusionmatrix)

sb.heatmap(confusionmatrix, annot=True, cmap="Blues")
plt.xlabel('Prediction')
plt.ylabel('Actual')





########### DECISION TREE MODEL

clf = DecisionTreeClassifier(min_samples_split = 3, min_samples_leaf = 2, max_depth = 8)

clf = clf.fit(train,labeltrain)

testpred = clf.predict(test)


print("Accuracy:", metrics.accuracy_score(labeltest, list(testpred)))
treecorrect = list(testpred == labeltest)
sum(treecorrect)/len(treecorrect)



actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(testpred, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
type(confusionmatrix)

sb.heatmap(confusionmatrix, annot=True, cmap="Blues")
plt.xlabel('Prediction')
plt.ylabel('Actual')




dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names = dtmdf.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('CICEROTREE2.png')
Image(graph.create_png())



############### SVM MODEL


svmmodel = SVM.SVC(kernel = 'linear')

svmmodel.fit(train, labeltrain)

svmpredictions = svmmodel.predict(test)

svmcorrect = list(svmpredictions == labeltest)

sum(svmcorrect) / len(labeltest)



actual = pd.Series(labeltest, dtype= "category")
pred = pd.Series(svmpredictions, dtype= "category")
confusionmatrix = pd.crosstab(actual, pred)
type(confusionmatrix)

sb.heatmap(confusionmatrix, annot=True, cmap="Blues")
plt.xlabel('Prediction')
plt.ylabel('Actual')
