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
from sklearn.ensemble import RandomForestClassifier
from textatistic import Textatistic
import spacy
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dateutil import parser
import datetime, time
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


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


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score['compound']


# read input file into dataset object
dataset = pandas.read_csv('ks-projects-201801.csv')

# drop columns we don't need
# dataset = dataset.drop('ID', axis=1)
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
arr = dataset.iloc[:, 4]
arr[arr=='failed'] = 0
arr[arr=='canceled'] = 0
arr[arr=='suspended'] = 0
arr[arr=='undefined'] = 0
arr[arr=='successful'] = 1

# convert datetime into scaled down value in seconds
arr = dataset.iloc[:, 2]
datetime_arr = []

for i, val in enumerate(arr):
    dt = parser.parse(val)
    datetime_in_seconds = time.mktime(dt.timetuple())
    scaled_down_datetime_in_seconds = datetime_in_seconds / 2000000000
    datetime_arr.append(scaled_down_datetime_in_seconds)

dataset['updated_datetime'] = datetime_arr

# convert launched date into scaled down value in seconds
arr = dataset.iloc[:, 3]
launched_arr = []

for i, val in enumerate(arr):
    launched_dt = parser.parse(val)
    launched_datetime_in_seconds = time.mktime(launched_dt.timetuple())
    scaled_down_launched_datetime_in_seconds = launched_datetime_in_seconds / 2000000000
    launched_arr.append(scaled_down_launched_datetime_in_seconds)

dataset['updated_launched'] = launched_arr

# convert titles into positivity score
arr = dataset.iloc[:, 1]
positivity_arr = []
bool_positivity_arr = []
flesch_kinaid_arr = []
bool_flesch_arr = []

for i, val in enumerate(arr):
    if i%2000 == 0:
        print(i)
    if type(val) is float:
        bool_positivity_arr.append(0)
        bool_flesch_arr.append(0)
        flesch_kinaid_arr.append(0)
        positivity_arr.append(sentiment_analyzer_scores('No name'))
    elif type(val) is str:
        if (sentiment_analyzer_scores(val) > -0.05 and sentiment_analyzer_scores(val) < 0.05):
            bool_positivity_arr.append(0)
        elif (sentiment_analyzer_scores(val) >= 0.05):
            bool_positivity_arr.append(1)
        elif (sentiment_analyzer_scores(val) <= -0.05):
            bool_positivity_arr.append(-1)

        positivity_arr.append(sentiment_analyzer_scores(val))
        val = val + '..'
        flesch_kinaid_arr.append(Textatistic(val).fleschkincaid_score)

        if (Textatistic(val).fleschkincaid_score < 4):
            bool_flesch_arr.append(-1)
        elif (Textatistic(val).fleschkincaid_score >= 4 and Textatistic(val).fleschkincaid_score <= 10):
            bool_flesch_arr.append(0)
        elif (Textatistic(val).fleschkincaid_score > 10):
            bool_flesch_arr.append(1)

dataset['positivity'] = positivity_arr
dataset['flesch_kinaid'] = flesch_kinaid_arr
dataset['bool_positivity'] = bool_positivity_arr
dataset['bool_flesch'] = bool_flesch_arr


print('made it here')


# trying to plot % of success campaigns based on positivity score

main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(50,200,50)'})
# Replacing unknown value to nan
dataset['bool_flesch'] = dataset['bool_flesch'].replace('N,0"', np.nan)

data = []
total_expected_values = []
annotations = []
shapes = []

rate_success_positivity = dataset[dataset['state'] == 1].groupby(['bool_flesch']).count()['ID']\
                / dataset.groupby(['bool_flesch']).count()['ID'] * 100
rate_failed_positivity = dataset[dataset['state'] == 0].groupby(['bool_flesch']).count()['ID']\
                / dataset.groupby(['bool_flesch']).count()['ID'] * 100
    
rate_success_positivity = rate_success_positivity.sort_values(ascending=False)
rate_failed_positivity = rate_failed_positivity.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_positivity.index,
        y=rate_success_positivity,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_positivity.index,
        y=rate_failed_positivity,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by positivity',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations,
    shapes=shapes
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='main_pos')







# print('X columns:')
# print(list(dataset.drop(['state', 'name', 'deadline', 'launched', 'positivity', 'backers', 'usd_pledged_real'], axis=1).columns.values))

# # split data into training and testing sets
# # http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/
# X = dataset.drop(['state', 'name', 'deadline', 'launched', 'backers', 'usd_pledged_real', 'flesch_kinaid', 'bool_positivity'], axis=1).values
# y = dataset[['state']].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# # flatten labels
# y_train_updated = y_train.reshape((338275,))
# y_train_updated = y_train_updated.astype('int')
# y_test_updated = y_test.reshape((37587,))
# y_test_updated = y_test_updated.astype('int')

# # run and fit the linearsvc svm model
# clf = LinearSVC(random_state=0, tol=1e-5)
# clf.fit(X_train, y_train_updated)

# # get the model accuracy
# score_test = clf.score(X_test, y_test_updated)
# print('accuracy:')
# print(score_test)

# # get the predictions
# pred = clf.predict(X_test)

# # get feature importance
# # https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
# rnd_clf.fit(X_train, y_train_updated)

# # plotting feature importance values
# features = list(dataset.drop(['state', 'name', 'deadline', 'launched', 'backers', 'usd_pledged_real', 'flesch_kinaid', 'bool_positivity'], axis=1).columns.values)
# importances = rnd_clf.feature_importances_
# indices = np.argsort(importances)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()

# # get correlation
# # correlation = matthews_corrcoef(y_test_updated, pred)
# # print('CORRELATION: {}'.format(correlation))

# # print('Writting Datasets to CSVs:')
# # dataset_1.to_csv('test_data.csv', encoding='utf-8', index=False)
# # dataset_2.to_csv('train_data.csv', encoding='utf-8', index=False)
