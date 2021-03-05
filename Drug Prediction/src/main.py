#Name : Aditya Varma Vetukuri
#Gnumber : G01213246

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import f1_score

#Naive Bayes Classifier

class NaiveBayes_Classifier:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self.lab = y
        n_classes = len(self._classes)
        self.dict_features_count = dict()
        self.dict_features_positivecount = dict()
        self.dict_features_negativecount = dict()
        for label,i in tqdm(enumerate(X)):
            for j in range(0,len(i)):
                if(i[j] == self._classes[1]):
                    if j not in self.dict_features_count:
                        self.dict_features_count[j] = 1
                    else:
                        self.dict_features_count[j] += 1
                    if(y[label] == 0):
                        if j not in self.dict_features_negativecount:
                            self.dict_features_negativecount[j] = 1
                        else:
                            self.dict_features_negativecount[j] += 1
                    else:
                        if j not in self.dict_features_positivecount:
                            self.dict_features_positivecount[j] = 1
                        else:
                            self.dict_features_positivecount[j] += 1
    def predict(self, X):
        y_pred = []
        positiveprob = 1
        negativeprob = 1
        for x in tqdm(X):
            for j in range(0, len(x)):
                if(x[j] == 1):
                    if j in self.dict_features_count:
                        if j not in self.dict_features_positivecount:
                            self.dict_features_positivecount[j] = 1
                        if j not in self.dict_features_negativecount:
                            self.dict_features_negativecount[j] = 1
                        positiveprob *= (self.dict_features_positivecount[j] / self.dict_features_count[j])
                        negativeprob *= (self.dict_features_negativecount[j] / self.dict_features_count[j])

            if (positiveprob > negativeprob):
                y_pred.append(1)
            else:
                y_pred.append(0)
            positiveprob = 1
            negativeprob = 1
        return y_pred

#Gaussian Naive Bayes Classifier

class Naive_Bayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[y == c]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
    def predict(self, X):
        y_pred = [self.each_class_predict(x) for x in X]
        return np.array(y_pred)

    def each_class_predict(self, x):
        posteriors = []

        for idx, c in tqdm(enumerate(self._classes)):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self.prob_formula(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)


        return self._classes[np.argmax(posteriors)]

    def prob_formula(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

Lis = []
Lis2 = []

#Getting Class Labels
def get_labels(fileName):
    Labels = []
    for line in open(fileName):
        Labels.append(int(line[0].rstrip('\n')))

    return Labels
#Getting feautures of each drug
def get_features(fileName):
    features = []
    for line in open(fileName):
        features.append(line[2:].rstrip('\n'))
    return features
#Getting Test data
def  get_testData(fileName):
    test_features = []
    for line in open(fileName):
        test_features.append(line[:-1].rstrip('\n'))
    return test_features
#Removing ' ' and making lists of size 3 to fit into our CSR_matrix
def preprocess(lis):
    sett = []
    for i in range(0, len(lis)):
        lis[i] = lis[i].split(' ')
        for a in lis[i]:
            if (a != ''):
                sett.append([i, int(a), float(1)])
    return sett
#Converting data into CSR_Matrix
def convert_matrix(features):
    rows,cols,vals = zip(*features)
    csr_feature_matrix = csr_matrix((vals , (rows, cols)))
    csr_feature_matrix_array = csr_feature_matrix.toarray()
    return csr_feature_matrix_array
#Feature Selection Pipeline
def feature_selection_pipeline(X_train,Y_train,X_test, Value):
    if(Value == '1'):
        bestfeatures = SelectKBest(score_func=chi2, k = 5)
        bestfeatures.fit(X_train, Y_train)
        X_train = bestfeatures.transform(X_train)
        X_test = bestfeatures.transform(X_test)
        return X_train,X_test
    if(Value == '2'):
        sm = SMOTE(random_state = 33)
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
        scaler = StandardScaler()
        scaler.fit(X_train,Y_train)
        X = scaler.transform(X_train)
        Y = scaler.transform(X_test)
        pca = PCA(600)
        pca.fit(X)
        X = pca.transform(X)
        Y = pca.transform(Y)
        return X,Y
    if(Value == '3'):
        tsvd = TruncatedSVD(n_components=5000)
        X = tsvd.fit(X_train).transform(X_train)
        Y = tsvd.transform(X_test)
        return X,Y
#Classification Model Pipeline
def model_pipeline(X_train,Y_train,X_test,Y_test,Value):
    if(Value == '1'):
        model = DecisionTreeClassifier(class_weight = "balanced")
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return Y_test
    elif(Value == '2'):
        model = Naive_Bayes()
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return Y_test
    elif(Value == '3'):
        model = NaiveBayes_Classifier()
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return Y_test
    elif(Value == '4'):
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),random_state=1)
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return Y_test
    elif(Value == '5'):
        model = RandomForestClassifier()
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return Y_test
#Accuracy function
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


filename = "/Users/adityavarma/Downloads/1582371142_761199_train_drugs.dat"
Lis2 = get_features(filename)
Lis = get_labels(filename)
preprocessed_features = preprocess(Lis2)
test_lis = get_testData("/Users/adityavarma/Downloads/1582371142_7687795_test.dat")
preprocessed_test_features = preprocess(test_lis)
X_train = convert_matrix(preprocessed_features)
X_test = convert_matrix(preprocessed_test_features)
Y_train = Lis
Y_test = []
featureselection_value = input("Enter a Feature Selection Value \n1: Select K Best\n2:Principal Component Analysis\n3:Truncated SVD\n")

#SMOTE SAMPLING TECHNIQUE

# oversample = SMOTE()
# X_train, Y_train = oversample.fit_resample(X_train, Lis)

#test-train split
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=1)

X_train,X_test = feature_selection_pipeline(X_train,Y_train,X_test,featureselection_value)
model_value = input("Enter a Value:  \n1 : Decision Tree \n2 : GaussianiNaiveBayes \n3 : NaiveBayesFrom Scratch \n4: NeuralNetworks \n5: RandomForest\n")\

#KFOLD CROSS VALIDATION IMPLEMENTATION
################################################################################################
# kf = KFold(n_splits=2)
# kf.get_n_splits(X_train)
# F1_scores = []
# for train_index, test_index in kf.split(X_train):
#     X_train, X_test = X_train[train_index], X_train[test_index]
#     y_train, y_test = Y_train[train_index], Y_train[test_index]
#
#     print("Testing all classifier models \n")
#     model = DecisionTreeClassifier(class_weight="balanced")
#     model.fit(X_train, y_train)
#     Y_pred = model.predict(X_test)
#     F1_scores.append(f1_score(Y_pred,y_test))
#
#     model = RandomForestClassifier(class_weight="balanced")
#     model.fit(X_train, y_train)
#     Y_pred = model.predict(X_test)
#     F1_scores.append(f1_score(Y_pred, y_test))
#
#     model = Naive_Bayes()
#     model.fit(X_train, y_train)
#     Y_pred = model.predict(X_test)
#     F1_scores.append(f1_score(Y_pred, y_test))
#
#     model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#     model.fit(X_train, Y_train)
#     Y_pred = model.predict(X_test)
#     F1_scores.append(f1_score(Y_pred, y_test))
#
# print(F1_scores)

#model pipeline
y_pred = model_pipeline(X_train,Y_train,X_test,Y_test,model_value)

f = open("dtreefinal.txt","w")
for i in y_pred:
    f.write("%s\n" % i)
f.close()









