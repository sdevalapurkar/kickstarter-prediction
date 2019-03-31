import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import math
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import svm, datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA

def balancing_info(df):
    succ = 0
    fail = 0

    for row in df:
        if row[0] == 'failed' or row[0] == 'canceled':
            fail += 1
        elif row[0] == 'successful':
            succ += 1

    return succ, fail


def get_list_of_main_categories(df):
    list_of_categories = []

    for row in df:
        if row[0] not in list_of_categories:
            list_of_categories.append(row[0])

    return list_of_categories


def get_list_of_countries(df):
    list_of_countries = []

    for row in df:
        if row[0] not in list_of_countries:
            list_of_countries.append(row[0])

    return list_of_countries


def convertToNumber (s):
    return int.from_bytes(s.encode(), 'little')


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score['compound']


# read input file into dataset object
dataset = pandas.read_csv('ks-projects-201801.csv')


# new_dataset_sauce = dataset.drop('state', axis=1)
# new_target_sauce = dataset[['state']]
# numpy_dataset = new_dataset_sauce.to_numpy()
# new_target_sauce = new_target_sauce.to_numpy()
# new_target_sauce = new_target_sauce.reshape((378661,))

# drop columns we don't need
dataset = dataset.drop('ID', axis=1)
dataset = dataset.drop('category', axis=1)
dataset = dataset.drop('goal', axis=1)
dataset = dataset.drop('pledged', axis=1)
dataset = dataset.drop('usd pledged', axis=1)
dataset = dataset.drop('currency', axis=1)

# handle nominal categorical data
# http://benalexkeen.com/mapping-categorical-data-in-pandas/
list_of_main_categories = get_list_of_main_categories(dataset[['main_category']].values)
dataset.main_category.astype("category", categories=list_of_main_categories).cat.codes
dataset = pandas.get_dummies(dataset, columns=['main_category'])

list_of_countries = get_list_of_countries(dataset[['country']].values)
dataset.country.astype("category", categories=list_of_countries).cat.codes
dataset = pandas.get_dummies(dataset, columns=['country'])

# remove currently live campaigns
dataset = dataset[dataset.state != 'live']

# turn state of campaign into boolean value
arr = dataset.iloc[:, 3]
arr[arr=='failed'] = 0
arr[arr=='canceled'] = 0
arr[arr=='suspended'] = 0
arr[arr=='undefined'] = 0
arr[arr=='successful'] = 1

# convert titles into positivity score
arr = dataset.iloc[:, 0]
positivity_arr = []
for i, val in enumerate(arr):
    if type(val) is float:
        positivity_arr.append(sentiment_analyzer_scores('No name'))
    elif type(val) is str:
        positivity_arr.append(sentiment_analyzer_scores(val))
dataset['positivity'] = positivity_arr

print('X columns:')
print(list(dataset.drop(['state', 'name', 'deadline', 'launched', 'positivity', 'backers', 'usd_pledged_real'], axis=1).columns.values))

# split data into training and testing sets
# http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/
X = dataset.drop(['state', 'name', 'deadline', 'launched'], axis=1).values
y = dataset[['state']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# flatten labels
y_train_updated = y_train.reshape((281896,))
y_train_updated = y_train_updated.astype('int')
y_test_updated = y_test.reshape((93966,))
y_test_updated = y_test_updated.astype('int')








# feature importance :) works!
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train_updated)

for name, importance in zip(list(dataset.drop(['state', 'name', 'deadline', 'launched'], axis=1).columns.values), rnd_clf.feature_importances_):
    print(name, "=", importance)






# plotting feature importances
features = list(dataset.drop(['state', 'name', 'deadline', 'launched'], axis=1).columns.values)
importances = rnd_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()














# run and fit the linearsvc svm model
# clf = LinearSVC(random_state=0, tol=1e-5)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train_updated)

# # get the model accuracy
# score_test = clf.score(X_test, y_test_updated)

# print('score test')
# print(score_test)

# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel

# print(X_train.shape)
# clf = ExtraTreesClassifier(n_estimators=50)
# clf = clf.fit(X_train, y_train)
# print(clf.feature_importances_)
# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X_train)
# print(X_new.shape)

# # feature extraction using logistic regression RFE
# model = LogisticRegression()
# rfe = RFE(model, 3)
# fit = rfe.fit(X_train, y_train_updated)

# print("Num Features:")
# print(fit.n_features_)
# print("Selected Features:")
# print(fit.support_)
# print("Feature Ranking:")
# print(fit.ranking_)


# plot.scatter(xTrain, yTrain, color = 'red')
# plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
# plot.title('Success vs Positivity (Training set)')
# plot.xlabel('Positivity')
# plot.ylabel('Success')
# plot.show()

# print('Writting Datasets to CSVs:')
# dataset_1.to_csv('test_data.csv', encoding='utf-8', index=False)
# dataset_2.to_csv('train_data.csv', encoding='utf-8', index=False)
