import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import math
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

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

print(dataset.head())

# drop columns we don't need
dataset = dataset.drop('ID', axis=1)
dataset = dataset.drop('category', axis=1)
dataset = dataset.drop('goal', axis=1)
dataset = dataset.drop('pledged', axis=1)
dataset = dataset.drop('usd pledged', axis=1)
dataset = dataset.drop('currency', axis=1)

# handle nominal categorical data
list_of_main_categories = get_list_of_main_categories(dataset[['main_category']].values)
dataset.main_category.astype("category", categories=list_of_main_categories).cat.codes
dataset = pandas.get_dummies(dataset, columns=['main_category'])

list_of_countries = get_list_of_countries(dataset[['country']].values)
dataset.country.astype("category", categories=list_of_countries).cat.codes
dataset = pandas.get_dummies(dataset, columns=['country'])

print(dataset.head())

# remove currently live campaigns
dataset = dataset[dataset.state != 'live']

print(dataset)

# turn state of campaign into boolean value
arr = dataset.iloc[:, 3]
arr[arr=='failed'] = 0
arr[arr=='canceled'] = 0
arr[arr=='suspended'] = 0
arr[arr=='undefined'] = 0
arr[arr=='successful'] = 1

# split data into training and testing sets
X = dataset.drop(['state', 'name', 'deadline', 'launched', 'backers'], axis=1).values
y = dataset[['state']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# flatten labels
y_train_updated = y_train.reshape((281896,))
y_train_updated = y_train_updated.astype('int')
y_test_updated = y_test.reshape((93966,))
y_test_updated = y_test_updated.astype('int')

# run and fit the linearsvc svm model
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train_updated)

# clf.fit(X_train, y_train.values.ravel())
score_train = clf.score(X_train, y_train_updated)
score_test = clf.score(X_test, y_test_updated)

print('score train')
print(score_train)
print('score test')
print(score_test)

# # extract important features from dataset
# dataset = dataset[
#     [
#         'ID',
#         'state',
#         'currency',
#         'deadline',
#         'launched',
#         'name',
#         'country',
#         'main_category',
#         'backers',
#         'goal',
#         'usd pledged'
#     ]
# ]

# # normalize main category values as floats
# cats_dict_vectors = {
#     'Publishing': 1/15,
#     'Film & Video': 2/15,
#     'Music': 3/15,
#     'Food': 4/15,
#     'Design': 5/15,
#     'Crafts': 6/15,
#     'Games': 7/15,
#     'Comics': 8/15,
#     'Fashion': 9/15,
#     'Theater': 10/15,
#     'Art': 11/15,
#     'Photography': 12/15,
#     'Technology': 13/15,
#     'Dance': 14/15,
#     'Journalism': 15/15,
# }

# arr = dataset.iloc[:, 7]
# arr[arr=='Publishing'] = 1/15
# arr[arr=='Film & Video'] = 2/15
# arr[arr=='Music'] = 3/15
# arr[arr=='Food'] = 4/15
# arr[arr=='Design'] = 5/15
# arr[arr=='Crafts'] = 6/15
# arr[arr=='Games'] = 7/15
# arr[arr=='Comics'] = 8/15
# arr[arr=='Fashion'] = 9/15
# arr[arr=='Theater'] = 10/15
# arr[arr=='Art'] = 11/15
# arr[arr=='Photography'] = 12/15
# arr[arr=='Technology'] = 13/15
# arr[arr=='Dance'] = 14/15
# arr[arr=='Journalism'] = 15/15

# # normalize country values as floats
# country_dict_vectors = {
#     'GB': 1/21,
#     'US': 2/21,
#     'CA': 3/21,
#     'AU': 4/21,
#     'NO': 5/21,
#     'IT': 6/21,
#     'DE': 7/21,
#     'IE': 8/21,
#     'MX': 9/21,
#     'ES': 10/21,
#     'SE': 11/21,
#     'FR': 12/21,
#     'NL': 13/21,
#     'NZ': 14/21,
#     'CH': 15/21,
#     'AT': 16/21,
#     'DK': 17/21,
#     'BE': 18/21,
#     'HK': 19/21,
#     'SG': 20/21,
#     'JP': 21/21,
# }

# arr = dataset.iloc[:, 6]
# arr[arr=='GB'] = 1/21
# arr[arr=='US'] = 2/21
# arr[arr=='CA'] = 3/21
# arr[arr=='AU'] = 4/21
# arr[arr=='NO'] = 5/21
# arr[arr=='IT'] = 6/21
# arr[arr=='DE'] = 7/21
# arr[arr=='IE'] = 8/21
# arr[arr=='MX'] = 9/21
# arr[arr=='ES'] = 10/21
# arr[arr=='SE'] = 11/21
# arr[arr=='FR'] = 12/21
# arr[arr=='NL'] = 13/21
# arr[arr=='NZ'] = 14/21
# arr[arr=='CH'] = 15/21
# arr[arr=='AT'] = 16/21
# arr[arr=='DK'] = 17/21
# arr[arr=='BE'] = 18/21
# arr[arr=='HK'] = 19/21
# arr[arr=='SG'] = 20/21
# arr[arr=='JP'] = 21/21

# # convert titles into positivity score
# arr = dataset.iloc[:, 5]
# positivity_arr = []
# for i, val in enumerate(arr):
#     if i % 2000 == 0:
#         print(i)
#     positivity_arr.append(sentiment_analyzer_scores(val))

# dataset['positivity'] = positivity_arr

# print('new positivity column:')
# print(dataset[['positivity']].values)

# print(dataset)

# x = dataset.iloc[:, 7:12].values
# y = dataset.iloc[:, 1].values

# print(x)

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state=1)
# linearRegressor = LinearRegression()
# linearRegressor.fit(xTrain, yTrain)

# for idx, col_name in enumerate(xTrain.columns):
#     print("The coefficient for {} is {}".format(col_name, linearRegressor.coef_[0][idx]))


# score = linearRegressor.score(xTrain, yTrain)

# print('Linear regression score for train:')
# print(score)

# score_test = linearRegressor.score(xTest, yTest)

# print('Linear regression score for test:')
# print(score_test)

# yPrediction = linearRegressor.predict(xTest)

# plot.scatter(xTrain, yTrain, color = 'red')
# plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
# plot.title('Success vs Positivity (Training set)')
# plot.xlabel('Positivity')
# plot.ylabel('Success')
# plot.show()

# print('Writting Datasets to CSVs:')
# dataset_1.to_csv('test_data.csv', encoding='utf-8', index=False)
# dataset_2.to_csv('train_data.csv', encoding='utf-8', index=False)
