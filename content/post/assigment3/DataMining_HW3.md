---
data: 2020-04-21
title: Assigment3
---

```python
import numpy as np
import nltk 
import re
import math
from random import choice
import time
import _pickle as cpickle
import os

def get_data(x_file, y_file):
    x_file_obj = open(x_file, 'r', encoding="utf8")
    x = []
    for line in x_file_obj:
        x.append(line)
    y = np.loadtxt(y_file)
    return x, y

def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts
        
def train_model(x, y, modi, smooth=False):
    vocab_dict = {}
    word_count_labels = {}
    theta_words = {}
    len_of_examples = {}
    labels_count = {}
    log_labels_prior = {}
    if(modi):
        file_train_name = "objs/train_obj_modi"
    else:
        file_train_name = "objs/train_obj"
    if os.path.exists(file_train_name):
        fileObj = open(file_train_name, 'rb')
        arr = cpickle.load(fileObj)
        log_labels_prior = arr[0] 
        theta_words = arr[1]
        vocab_dict = arr[2]
        labels_count = arr[3]
        return log_labels_prior, theta_words, vocab_dict, labels_count
    
    N1 = len(y)
    for i in range(1,11):
        if i == 5 or i == 6: continue
        word_count_labels[i] = {}
        theta_words[i] = {}
        len_of_examples[i] = 0

    i = 0
    for line in x:
#         tokens = nltk.word_tokenize(line)
        line = line.lower()
        tokens = re.split("\W+", line)
        word_count = get_word_counts(tokens)
        for word, count in word_count.items():
            if word not in vocab_dict:
                vocab_dict[word] = 0
            if word not in word_count_labels[y[i]]:
                word_count_labels[y[i]][word] = 0.0
            word_count_labels[y[i]][word] += count
        if y[i] not in labels_count:
            labels_count[y[i]] = 0.0
        labels_count[y[i]] += 1
        len_of_examples[y[i]] += len(tokens)
        i+=1
        
    for j in range(1,11):
        if j ==5 or j == 6: continue
        log_labels_prior[j] = math.log(labels_count[j] / N1)
        
    for j in range(1,11):
        if j == 5 or j == 6: continue
        for word in vocab_dict:
            if smooth:
                temp = (int(word_count_labels[j].get(word,0.0)) + 1) / (int(len_of_examples[j]) + int(len(vocab_dict)))
            else:
                temp = (int(word_count_labels[j].get(word,0.0))) / (int(len_of_examples[j]))
            theta_words[j][word] = temp   
    fileObj = open(file_train_name,'wb')    
    arr = [log_labels_prior, theta_words, vocab_dict, labels_count]     
    cpickle.dump(arr, fileObj)
    return log_labels_prior, theta_words, vocab_dict, labels_count

""" accuracy train = 78.15, test = 40.33% """    
def naive_bayes_classifier(x, log_labels_prior, theta_words, vocab_dict, modi, train):
    if(modi):
        if(train):
            file_nv_c_name = "objs/nv_c_res_modi_train"
        else:
            file_nv_c_name = "objs/nv_c_res_modi_test"
    else:
        if(train):
            file_nv_c_name = "objs/nv_c_res_train"
        else:
            file_nv_c_name = "objs/nv_c_res_test"
    if os.path.exists(file_nv_c_name):
        fileObj = open(file_nv_c_name, 'rb')
        result = cpickle.load(fileObj)
        return result
    
    result = []
    for line in x:
        #tokens = nltk.word_tokenize(line)
        #line = line.lower()
        tokens = re.split("\W+", line)
        class_label_score = {1:0, 2:0, 3:0, 4:0, 7:0, 8:0, 9:0, 10:0}
        word_count = get_word_counts(tokens)
        for word, count in word_count.items():
            if word not in vocab_dict: continue
            for i in range(1,11):
                if i== 5 or i == 6: continue
                class_label_score[i] += (math.log(theta_words[i][word]))
            
        for i in range(1,11):
            if i == 5 or i == 6:continue
            class_label_score[i] += float((log_labels_prior[i]))
        #print(class_label_score)
        result.append(max(class_label_score, key = class_label_score.get))
        
    fileObj = open(file_nv_c_name,'wb')    
    cpickle.dump(result, fileObj)    
    return result

def prediction(x, log_labels_prior, theta_words, vocab_dict, labels_count, modi, train):
    result = naive_bayes_classifier(x, log_labels_prior, theta_words, vocab_dict, modi, train)
    return np.array(result)

def find_accuracy(y, result):
    N = len(y)
    sum = 0
    for i in range(0,N):
        if y[i] == result[i]:
            sum += 1
    return sum/N
    
def create_conf_matrix(expected, predicted):
    conf_mat = np.zeros((11,11), dtype = int)
    N = len(expected)
    for i in range(0,N):
        if int(expected[i]) == 5 or int(expected[i]) == 6: continue
        conf_mat[int(expected[i])][int(predicted[i])] +=1
    
    conf_mat = np.delete(conf_mat, 0, 0)
    conf_mat = np.delete(conf_mat, 4, 0)
    conf_mat = np.delete(conf_mat, 4, 0)
    conf_mat = np.delete(conf_mat, 0, 1)
    conf_mat = np.delete(conf_mat, 4, 1)
    conf_mat = np.delete(conf_mat, 4, 1)
    return conf_mat
```

## Baseline


```python
x_train_file = "imdb_rev/imdb_train_text_modified.txt"
y_train_file = "imdb_rev/imdb_train_labels.txt"
x_test_file = "imdb_rev/imdb_test_text_modified.txt"
y_test_file = "imdb_rev/imdb_test_labels.txt"
start_time = time.time()
X_train, Y_train = get_data(x_train_file, y_train_file)
X_test, Y_test = get_data(x_test_file, y_test_file)
log_labels_prior, theta_words, vocab_dict, labels_count = train_model(X_train, Y_train, 1)
"""
classifier:
1.-> random classifier
2. -> majority classifier
3. -> naive bayes classifier
"""
result = prediction(X_train, log_labels_prior,theta_words, vocab_dict, labels_count, 1, 1) #classifier, modi, train
accuracy = find_accuracy(Y_train, result)
print(("The accuracy of training set is :{0:.4f}".format(accuracy)))
test_result = prediction(X_test, log_labels_prior,theta_words, vocab_dict, labels_count, 1, 1) #classifier, modi, train
test_accuracy = find_accuracy(Y_test, test_result)
print(("The accuracy of test set is :{0:.4f}".format(test_accuracy)))
print("--- %s seconds ---" % (time.time() - start_time))
```

    The accuracy of training set is :0.6953
    The accuracy of test set is :0.2954
    --- 0.33498287200927734 seconds ---


## Cross validation


```python
from sklearn.model_selection import KFold
import numpy as np
x_train_file = "imdb_rev/imdb_train_text_modified.txt"
y_train_file = "imdb_rev/imdb_train_labels.txt"
x_test_file = "imdb_rev/imdb_test_text_modified.txt"
y_test_file = "imdb_rev/imdb_test_labels.txt"
start_time = time.time()
X_train, Y_train = get_data(x_train_file, y_train_file)
X_test, Y_test = get_data(x_test_file, y_test_file)

kf = KFold(n_splits=5)
kf.get_n_splits(X_train)
X_train, Y_train = np.array(X_train), np.array(Y_train)
for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    print(train_index)
    print(test_index)
    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
    Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
    log_labels_prior, theta_words, vocab_dict, labels_count = train_model(X_train_cv, Y_train_cv, 1)
    result = prediction(X_test_cv, log_labels_prior, theta_words, vocab_dict, labels_count, 1, 1) #classifier, modi, train
    accuracy = find_accuracy(Y_test_cv, result)
    print("The accuracy of current cross validation is :{0:.4f}".format(accuracy))
```

    [ 5000  5001  5002 ... 24997 24998 24999]
    [   0    1    2 ... 4997 4998 4999]
    The accuracy of current cross validation is :0.6922
    [    0     1     2 ... 24997 24998 24999]
    [5000 5001 5002 ... 9997 9998 9999]
    The accuracy of current cross validation is :0.2786
    [    0     1     2 ... 24997 24998 24999]
    [10000 10001 10002 ... 14997 14998 14999]
    The accuracy of current cross validation is :0.1582
    [    0     1     2 ... 24997 24998 24999]
    [15000 15001 15002 ... 19997 19998 19999]
    The accuracy of current cross validation is :0.0340
    [    0     1     2 ... 19997 19998 19999]
    [20000 20001 20002 ... 24997 24998 24999]
    The accuracy of current cross validation is :0.0354


# Effect of Smooth


```python
x_train_file = "imdb_rev/imdb_train_text_modified.txt"
y_train_file = "imdb_rev/imdb_train_labels.txt"
x_test_file = "imdb_rev/imdb_test_text_modified.txt"
y_test_file = "imdb_rev/imdb_test_labels.txt"
start_time = time.time()
X_train, Y_train = get_data(x_train_file, y_train_file)
X_test, Y_test = get_data(x_test_file, y_test_file)
print('-----'*10)
log_labels_prior1, theta_words1, vocab_dict1, labels_count1 = train_model(X_train, Y_train, 1, False)
result = prediction(X_test, log_labels_prior1, theta_words1, vocab_dict1, labels_count1, 1, 1) #classifier, modi, train
accuracy = find_accuracy(Y_test, result)
print(("The accuracy of test set without smooth is :{0:.4f}".format(accuracy)))
print('-----'*10)
log_labels_prior2, theta_words2, vocab_dict2, labels_count2 = train_model(X_train, Y_train, 0, True)
test_result = prediction(X_test, log_labels_prior2, theta_words2, vocab_dict2, labels_count2, 0, 1) #classifier, modi, train
test_accuracy = find_accuracy(Y_test, test_result)
print(("The accuracy of test set with smooth is :{0:.4f}".format(test_accuracy)))
```

    --------------------------------------------------
    The accuracy of test set without smooth is :0.2954
    --------------------------------------------------
    The accuracy of test set with smooth is :0.3955



```python

```


```python

```
