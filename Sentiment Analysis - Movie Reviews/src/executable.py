#Aditya Varma Vetukuri
#GID : G01213246


import pandas as pd
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, accuracy_score
import numpy as np
from nltk.stem import WordNetLemmatizer
# Removing all special characters, numbers and punctuations
def cleaning(text):
    try:
        text = text.lower()
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\[.*?\]', '', text)
    except AttributeError:
        return 'TYPENOTSTRING'
    return text
#Stemming each word in the review and removing all the stop words and returning the review


def stemming(text):
    lem  = WordNetLemmatizer()
    try:
        after_stemming = []
        token_words = word_tokenize(text)
        for word in token_words:
            if lem.lemmatize(word) not in stopwords.words('english'):
                after_stemming.append(lem.lemmatize(word))
        return " ".join(after_stemming)
    except:
        return text
#Pre processing Training data and Testing data


def preprocessing(text):
    clean_data = []
    for i in tqdm(range(len(text))):
        data = cleaning(text[i])
        data = stemming(data)
        clean_data.append(data)
    return clean_data

#Vectorizing data with different Feature extractions


def vectorizer(train_corpus,test_corpus,vectorizer_type):
    if(vectorizer_type == 'COUNT'):
        count = CountVectorizer()
        cnt_corpus = count.fit_transform(train_corpus + test_corpus)
    if(vectorizer_type == 'BINARY'):
        binary = CountVectorizer(binary=True)
        bin_corpus = binary.fit_transform(train_corpus + test_corpus)
    if(vectorizer_type == 'TFIDF'):   #TF IDF VECTOR IMPLEMENTATION
        N = len(test_corpus)
        cnt_corpus = bin_corpus.toarray()
        tfidf = TfidfVectorizer()
        tfidf_corpus = tfidf.fit_transform(train_corpus + test_corpus)
        tfidf_corpus = np.zeros(cnt_corpus.shape)
        n_term_mapping = {i:np.sum(bin_corpus.getcol(0)) for i in range(bin_corpus.shape[1])}
        for x in zip(bin_corpus.nonzero()[0], bin_corpus.nonzero()[1]):
            tfidf_corpus[x[0]][x[1]] = cnt_corpus[x[0]][x[1]]*np.log10(N/n_term_mapping[x[1]])
        tfidf_corpus = csr_matrix(tfidf_corpus)


# Cross validation using k fold

def validation_data(train_format_data,train_data,k_cross_valid):
	valid_data=[]
	valid_format_data=[]
	train_data_1=[]
	train_format_data_1=[]
	leng= len(train_data)
	leng_per  = int(leng/k_cross_valid)
	for i in range(0,k_cross_valid):
		j=i*leng_per
		valid_data.append(train_data[j:(j+leng_per)])
		valid_format_data.append(train_format_data[j:(j+leng_per)])
		train_data_1.append(train_data[:j]+train_data[(j+leng_per):])
		train_format_data_1.append(train_format_data[:j]+train_format_data[(j+leng_per):])

	return valid_format_data,train_format_data_1,valid_data,train_data_1


#My implementation for knn classifier

def knnclassifier(X_train,X_test,y_train,y_test,k):
    distance = pairwise_distances(X_test, X_train, n_jobs=-1, metric='euclidean')
    for i in tqdm(range(distance.shape[0])):
        mapping = [(distance[i][j], y_train.iloc[j]) for j in range(len(distance[i]))]
        mapping.sort(key=lambda x: x[0])
        mapping = [_[1] for _ in mapping[:k]]
        s = np.sum(mapping)
        if s > 0:
            y_test_computed.append(1)
        else:
            y_test_computed.append(-1)




train_dataset = pd.read_csv("/Users/adityavarma/Downloads/train_file.dat", header=None, sep="\t")
fptr = open("/Users/adityavarma/Downloads/test_data.dat", 'r')
test_dataset = fptr.readlines()
fptr.close()
review_label = train_dataset[0]
review_text = train_dataset[1]
cleaned_data = preprocessing(review_text)
test_cleaneddataset = preprocessing(test_dataset)
print(test_dataset)
count = CountVectorizer()
vector_corpus = count.fit_transform(cleaned_data + test_cleaneddataset)
n = len(cleaned_data)
#K FOLDS IMPLEMENTATION
kfolds = 10
k_indices = [5,10,25,50,100]
valid_format_data,train_format_data,valid_data,train_data = validation_data(review_label,review_text,kfolds)
a = []
for i in k_indices:

    X = vectorizer.transform(train_data[i])

    Test = vectorizer.transform(valid_data[i])

    y_train = np.zeros(len(train_format_data[i]))

    for j in range(0, len(train_format_data[i])):
        y_train[j] = train_format_data[i][j]

    y_pred1 = knnclassifier(X_train,X_test,y_train,y_test,k)

    y = np.zeros(len(valid_format_data[i]))

    for j in range(0, len(valid_format_data[i])):
        y[j] = valid_format_data[i][j]

    a.append(accuracy_score(y, y_pred1))

# print
# np.mean(a)

# X_train, X_test, y_train, y_test = train_test_split(tfidf_corpus, review_label, test_size=0.20, random_state=1)
X_train = vector_corpus
X_test = vector_corpus
y_train = review_label
y_test = []
k = 14

y_test_computed = knnclassifier(X_train,X_test,y_train,y_test,k)


#writing the predicted values to the output file
fptr = open("./output30.dat", 'w')
for y in y_test_computed:
    if y > 0:
        fptr.write('+1\n')
    else:
        fptr.write('-1\n')
fptr.close()



