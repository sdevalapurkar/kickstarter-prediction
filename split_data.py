import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression

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

dataset = pandas.read_csv('ks-projects-201801.csv', low_memory = False)
print('Size of Original Dataset:')
print(str(dataset.shape) + '\n')

dataset = dataset[[
  'state',
  'ID',
  'name',
  'main_category',
  'currency',
  'deadline',
  'launched',
  'backers',
  'country',
  'goal',
  'usd pledged']]
print('Size of Dataset After Feature Extraction:')
print(str(dataset.shape) + '\n')

dataset = dataset.dropna(axis=0, how='any')
dataset = dataset[dataset.country != 'LU']
print("Size of Dataset After Removing Nulls:")
print(str(dataset.shape) + '\n')

# remove currently live campaigns
dataset = dataset[dataset.state != 'live']

arr = dataset.iloc[:, 0]
arr[arr=='failed'] = 0
arr[arr=='canceled'] = 0
arr[arr=='suspended'] = 0
arr[arr=='successful'] = 1

x = dataset.iloc[:, 9:11].values
y = dataset.iloc[:, 0].values

print('arr')
print(arr)

print('modified dataset')
print(dataset)

print('y is:')
print(y)

print('x is:')
print(x)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.99999, random_state = 0)

print(yTrain)

linearRegressor = LinearRegression()

linearRegressor.fit(xTrain, yTrain)
score = linearRegressor.score(xTrain, yTrain)
print(score)

yPrediction = linearRegressor.predict(xTest)

# plot.scatter(xTrain, yTrain, color = 'red')
# plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
# plot.title('Success vs KickstarterID (Training set)')
# plot.xlabel('KickstarterID')
# plot.ylabel('Success')
# plot.show()

# dataset_1, dataset_2 = train_test_split(dataset, test_size=0.5, random_state = 42)
# print('Size of Dataset 1:')
# print(str(dataset_1.shape) + '\n')
# print('Size of Dataset 2:')
# print(str(dataset_2.shape) + '\n')

# succ, fail = balancing_info(dataset_1[['state']].values)
# print('Balancing Info of Dataset 1:')
# print('Number Successful: ' + str(succ) + ', Number Fail: ' + str(fail))
# print('Percent Successful: ' + str(succ / (succ + fail)))
# print('===============================\n')

# succ, fail = balancing_info(dataset_2[['state']].values)
# print('Balancing Info of Dataset 2:')
# print('Number Successful: ' + str(succ) + ', Number Fail: ' + str(fail))
# print('Percent Successful: ' + str(succ / (succ + fail)))
# print('===============================\n')

# list_of_categories = get_list_of_main_categories(dataset_1[['main_category']].values)
# print('Getting Categories Info of Dataset 1:')
# print('list of categories mate')
# print(list_of_categories)
# print('===============================\n')

# list_of_categories = get_list_of_main_categories(dataset_2[['main_category']].values)
# print('Getting Categories Info of Dataset 2:')
# print('list of categories mate')
# print(list_of_categories)
# print('===============================\n')

# list_of_countries = get_list_of_countries(dataset_1[['country']].values)
# print('Getting Countries Info of Dataset 1:')
# print('list of countries mate')
# print(list_of_countries)
# print('===============================\n')

# list_of_countries = get_list_of_countries(dataset_2[['country']].values)
# print('Getting Countries Info of Dataset 2:')
# print('list of countries mate')
# print(list_of_countries)
# print('===============================\n')

# print(dataset_1['country'].values)

# print('Writting Datasets to CSVs:')
# dataset_1.to_csv('test_data.csv', encoding='utf-8', index=False)
# dataset_2.to_csv('train_data.csv', encoding='utf-8', index=False)
