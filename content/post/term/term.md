---
data: 2020-05-11
title: termproject
---

# Readme
i didn't done this on the github, because i just push it to colab.
and u can just open the link to run the whole project 
https://colab.research.google.com/drive/1sdzln9IUD7AUtkxJ1U5SPrgQbP9mhOWC#scrollTo=XH3vfalNVr3M

and all you have to do is open this link, and upload the dataset. 
then run all the code. 




```
import json
import re
from random import seed, randrange
from math import log
from sklearn.utils import shuffle
import pandas as pd

import nltk
nltk.download("stopwords")
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```


```

    /bin/bash: response-content-disposition=attachment%3B+filename%3Dbgg-13m-reviews.csv.zip: command not found
    <?xml version='1.0' encoding='UTF-8'?><Error><Code>AuthenticationRequired</Code><Message>Authentication required.</Message></Error>

**import the data set**

```
```




```
df = pd.read_csv('bgg-13m-reviews.csv',index_col=0)
df.head()
```

    /usr/local/lib/python3.6/dist-packages/numpy/lib/arraysetops.py:569: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>rating</th>
      <th>comment</th>
      <th>ID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sidehacker</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Varthlokkur</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dougthonus</td>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cypar7</td>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ssmooth</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>13</td>
      <td>Catan</td>
    </tr>
  </tbody>
</table>
</div>



**Data Preprocessing**


```
df = df.dropna()
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.637756e+06</td>
      <td>2.637756e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.852070e+00</td>
      <td>6.693990e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.775769e+00</td>
      <td>7.304447e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.401300e-45</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000e+00</td>
      <td>3.955000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000e+00</td>
      <td>3.126000e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000e+00</td>
      <td>1.296220e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+01</td>
      <td>2.724090e+05</td>
    </tr>
  </tbody>
</table>
</div>




```
reviews = df[['rating','comment']]
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>Currently, this sits on my list as my favorite...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>I know it says how many plays, but many, many ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.0</td>
      <td>i will never tire of this game.. Awesome</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0</td>
      <td>This is probably the best game I ever played. ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10.0</td>
      <td>Fantastic game. Got me hooked on games all ove...</td>
    </tr>
  </tbody>
</table>
</div>




```
reviews = shuffle(reviews)
reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7361740</th>
      <td>7.5</td>
      <td>Fun, thematic twist on the original.</td>
    </tr>
    <tr>
      <th>1537742</th>
      <td>7.0</td>
      <td>ESSEN 2014</td>
    </tr>
    <tr>
      <th>7835124</th>
      <td>6.5</td>
      <td>Seemed pretty good... for a pocket size micro ...</td>
    </tr>
    <tr>
      <th>2669846</th>
      <td>7.0</td>
      <td>Nice game and thoroughly enjoyable, although t...</td>
    </tr>
    <tr>
      <th>1743502</th>
      <td>9.0</td>
      <td>Very, very clever game.  Needs little space, a...</td>
    </tr>
  </tbody>
</table>
</div>



# Word Segmentation

We just claimed word segmentation is a hard problem, but in fact the segmentation part is quite easy! We’ll give a quick overview of the segmentation algorithm which assumes that we can evaluate a segmentation for optimality.

First, we note that by an elementary combinatorial argument there are 2^{n-1} segmentations of a word with n letters. To see this, imagine writing a segmentation of “homebuiltairplanes” with vertical bars separating the letters, as in “home | built | ai |  rpla | nes”. The maximum number of vertical bars we could place is one less than the number of letters, and every segmentation can be represented by describing which of the n-1 gaps contain vertical bars and which do not. We can hence count up all segmentations by counting up the number of ways to place the bars. A computer scientist should immediately recognize that these “bars” can represent digits in a binary number, and hence all binary numbers with n-1 digits correspond to valid segmentations, and these range from 0 to 2^{n-1} - 1, giving 2^{n-1} total numbers.


# first:
 we implement the “splitPairs” function, which accepts a string s as input and returns a list containing all possible split pairs (u,v) where s = uv. We achieve this by a simple list comprehension (gotta love list comprehensions!) combined with string slicing
# secondly
Note that the last entry in this list is crucial, because we may not want to segment the input word at all, and in the following we assume that “splitPairs” returns all of our possible choices of action. Next we define the “segment” function, which computes the optimal segmentation of a given word. In particular, we assume there is a global function called “wordSeqFitness” which reliably computes the fitness of a given sequence of words, with respect to whether or not it’s probably the correct segmentation.


```
def word_segmentation(str):
    def splitPairs(word):
        return [(word[:i+1], word[i+1:]) for i in range(len(word))]
    def segment(word):
        if not word: return []
        allSegmentations = [[first] + segment(rest)
                            for (first, rest) in splitPairs(word)]
        return max(allSegmentations, key = wordSeqFitness)
    w = re.sub('[^a-zA-Z]',' ', str).lower().split()   # Remove non-alphabetic characters
    sw = (nltk.corpus.stopwords.words('english'))      # Remove stopwords
    wd = [x for x in w if x not in sw]
    return wd
```


```
x = [word_segmentation(r) for r in reviews['comment']]
y = [round(r) for r in reviews['rating']]
```

# Divide Data
 


```
x_train, y_train, x_test, y_test = x, y, [], []
test_size = int(len(x)*0.01)

seed(1)

for _ in range(test_size):
    random_index = randrange(len(x_train))
    x_test.append(x_train.pop(random_index))
    y_test.append(y_train.pop(random_index))

print('Size of Train Set: ', len(x_train))
print('Size of Test Set: ', len(x_test))
```

    Size of Train Set:  2611379
    Size of Test Set:  26377


# Text Feature Extraction

n a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.

Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency:
tf-idf(t,d) = tf(t,d) * idf(t)


Using the TfidfTransformer’s default settings, TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False) the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as
idf(t) = log(1+n/1+df(t)) +1

where n is the total number of documents in the document set, and   df(t) is the number of documents in the document set that contain term . The resulting tf-idf vectors are then normalized by the Euclidean norm.

The following sections contain further explanations and examples that illustrate how the tf-idfs are computed exactly and how the tf-idfs computed in scikit-learn’s TfidfTransformer and TfidfVectorizer differ slightly from the standard textbook notation that defines the idf as
idf(t) = log(n/1+df(t))

In the TfidfTransformer and TfidfVectorizer with smooth_idf=False, the “1” count is added to the idf instead of the idf’s denominator:

idf(t) = log(n/df(t)) +1






```
# Get all the words in the training set non-repeatedly and record the index of each word
words_index_dict = {}
index = 0
for rating in x_train:
    for word in rating:
        if word in words_index_dict:
          continue
        else:
            words_index_dict[word]=index
            index+=1
```


```
tf={}
idf = [0 for _ in range(len(words_index_dict))]
for review_index, review in enumerate(x_train):
    review_counts = pd.value_counts(review)
    for word_index, word in enumerate(review):
        if word not in words_index_dict:
          continue
        else:
            tf[(review_index,words_index_dict[word])] = review_counts[word]/len(review)
            idf[words_index_dict[word]]+=1
temp = []
for cont in idf:
    temp.append(log(len(x_train)/(cont+1)))
tf = temp

```

# Algorithms

Naive Bayes Classifier. Naive Bayes is a kind of classifier which uses the Bayes Theorem. It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class. The class with the highest probability is considered as the most likely class.


![alt text](https://www.analyticsvidhya.com/wp-content/uploads/2015/09/Bayes_rule-300x172-300x172.png)

# Naive Bayes Classifier – Example

Because it is a supervied learning algorithm, we have a dataset with samples and labels accordingly. First, Naive Bayes Classifier calculates the probability of the classes. What does it mean exactly? Calculating that if we choose a random sample, what is the probability it belongs to a given class?

![alt text](https://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/naivebayes7.png)



After the training procedure we want to classify the new sample (circle with question mark). Then we have to consider the neighborhood of that sample. We can make predictions based on Bayes theorem.
![alt text](https://www.globalsoftwaresupport.com/wp-content/uploads/2018/02/naivebayes8.png)




```
class Naive_Bayes:
    def __init__(self, data):
        self.d = data.iloc[:, 1:]
        self.headers = self.d.columns.values.tolist()
        self.prior = np.zeros(len(self.d['Class'].unique()))
        self.conditional = {}
    
    def build(self):
        y_unique = self.d['Class'].unique()
        for i in range(0,len(y_unique)):
            self.prior[i]=(sum(self.d['Class']==y_unique[i])+1)/(len(self.d['Class'])+len(y_unique))
            
        for h in self.headers[:-1]:
            x_unique = list(set(self.d[h]))
            x_conditional = np.zeros((len(self.d['Class'].unique()),len(set(self.d[h]))))
            for j in range(0,len(y_unique)):
                for k in range(0,len(x_unique)):
                    x_conditional[j,k]=(self.d.loc[(self.d[h]==x_unique[k])&(self.d['Class']==y_unique[j]),].shape[0]+1)/(sum(self.d['Class']==y_unique[j])+len(x_unique))
        
            x_conditional = pd.DataFrame(x_conditional,columns=x_unique,index=y_unique)   
            self.conditional[h] = x_conditional       
        return self.prior, self.conditional
    
    def predict(self, X):
        classes = self.d['Class'].unique()
        ans = []
        for sample in X:
            prob = []
            for i in range(len(self.prior)):
                p_i = self.prior[i]
                for j, h in enumerate(self.headers[:-1]):
                    p_i *= self.conditional[h][sample[j]][i]
                prob.append(p_i)
            ans.append(classes[np.argmax(prob)])
        return ans
```

for key in tf:
    tf[key]*=idf[key[1]]
    tfidf=dict()
for rating in range(11):
    tfidf[rating]=[0 for _ in range(len(words_index_dict))]
for key, value in tf.items():
    label = y_train[key[0]]
    word_index = key[1]
    tfidf[label][word_index]+=value
for i in range(len(tfidf)):
    row_sum = sum(tfidf[i])
    tfidf[i]=[x/row_sum for x in tfidf[i]]



```
label_count = [0 for _ in range(11)] + [len(x_train)]
for rating in y_train:
    label_count[rating]+=1
```


```
def count_value(l:list):
    value_count={}
    for x in l:
        if x not in value_count:
            value_count[x]=0
        value_count[x]+=1
    return value_count

def predict(review):
    probability = []
    words_in_review_set = set(review)
    words_counts = count_value(review)
    for label in range(11):
        prob = 0
        for word in words_in_review_set:
            if word not in words_index_dict:
                continue
            prob+=log(tfidf[label][words_index_dict[word]]*words_counts[word]+1)
        prob *= label_count[label]/label_count[-1]
        probability.append(prob)
    return probability.index(max(probability))
```


```
correct = 0
for i in range(len(x_test)):
    ans = predict(x_test[i])
    print(ans)
    if predict(x_test[i]) == y_test[i]:
        correct+=1
accuracy = correct/len(x_test)
print("Accuracy = ", accuracy)
```

    Accuracy =  0.2648519543541722


# Reference
https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

https://jeremykun.com/2012/01/15/word-segmentation/



> 




