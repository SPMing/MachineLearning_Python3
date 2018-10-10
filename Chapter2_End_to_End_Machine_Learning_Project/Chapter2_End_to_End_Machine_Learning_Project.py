"""
Chapter2 End-to-End Machine Learning Project
"""
# Preset
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


####################
# Data Preparation #
####################

# Get the data
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()


import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)
housing = load_housing_data()
housing.head()
housing.info()
list(housing) # get column names
housing["ocean_proximity"].value_counts()
housing.describe()


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Random sampling
import numpy as np
np.random.seed(42)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
shuffled_indices = np.random.permutation(len(housing))
housing.iloc[np.random.permutation(len(housing))]
train_set, test_set = split_train_test(housing, 0.2)

# Sampling with by creating indexes
import hashlib
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column):   
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# scikit split data with designated test-train ratio
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["median_income"].hist()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].value_counts()
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) # if not <5, make = 5
housing["income_cat"].hist()

# Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Evaluate sampling performance
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Discover and visualize the data to gain insights
# Scatter plot with color map
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()

# Scatter plot on top of map image
import matplotlib.image as mpimg
PROJECT_ROOT_DIR = os.getcwd()
california_img = mpimg.imread(PROJECT_ROOT_DIR + '/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet")) # extent : scalars (left, right, bottom, top)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)
plt.legend(fontsize=16)
plt.show()


# Correlation
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000]) # xmin, xmax, ymin, ymax

# Data transformation
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# Deal with NaNs
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head() # display NaNs
sample_incomplete_rows # total_bedrooms NaN
# Option-1: Drop the whole row of obs with null values
opt1 = sample_incomplete_rows.dropna(subset=["total_bedrooms"])
# Option-2: Drop the attributes with null values
opt2 = sample_incomplete_rows.drop("total_bedrooms", axis=1)
# Option-3: Fill NaNs with median
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)

# Use Imputer for Option-3
from sklearn.preprocessing import Imputer
housing_num = housing.select_dtypes(include=[np.number]) # select numerical attributes only
imputer = Imputer(strategy="median")
imputer.fit(housing_num)
imputer.statistics_ == housing_num.median().values # compare median calculated by imputer with median()
# Transform the training set:
X = imputer.transform(housing_num)
X2 = imputer.fit_transform(housing_num) # fit and transform in one step
[X==X2]==False # check if results are the same
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))


# Categorical data
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

housing_cat = housing[['ocean_proximity']] # column vector [[...]]
housing_cat.head(10)
# 1: use LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat_encoded = encoder.fit_transform(housing_cat)
set(housing_cat_encoded)
encoder.classes_ # check class

# 2: use OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False) # OneHotEncoder only take 2D array
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

# 3: use LabelBinarizer to merge 1 and 2
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# Create a custom transformer to add extra attributes
from sklearn.base import BaseEstimator, TransformerMixin
# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
housing.values[:,3] # housing column 3
from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room: # if True
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, # np.c_ transpose from 1D to 2D
                         bedrooms_per_room]
        else: # if False
            return np.c_[X, rooms_per_household, population_per_household] 

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# Build a pipeline for preprocessing the numerical attributes
from sklearn.pipeline import Pipeline
# Normalization
from sklearn.preprocessing import StandardScaler # MinMaxScaler range from 0-1: (value-min)/(max-min)
                                                 # StandardScaler, standardization: (value-mean)/sd, less affected by outliers
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),      # 1: fill NaNs
        ('attribs_adder', CombinedAttributesAdder()), # 2: add extra attributes
        ('std_scaler', StandardScaler()),             # 3: standardization
    ])
# last estimator in pipeline must be transformer: have fit_transform() function
housing_num_tr = num_pipeline.fit_transform(housing_num)


# Pipeline for num and cat transformation, Union all previous transformations into a pipeline
# Roll back to original dataset
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

housing.info() # check attribute type
num_attribs = list(housing.select_dtypes(include=['number'])) # defined at imputer section
#cat_attribs = ['ocan_proximity']
cat_attribs = list(housing.select_dtypes(include=['object']))

# Create a class to select numerical or categorical columns 
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Create LabelBinarizer for Pipeline
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
      self.sparse_output = sparse_output
    def fit(self, X, y=None):
      self.enc = LabelBinarizer(sparse_output=self.sparse_output)
      self.enc.fit(X)
      return self
    def transform(self, X, y=None):
      return self.enc.transform(X)
  
#X = imputer.transform(housing_num)
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)), # num_attribs
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        #('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CustomLabelBinarizer()), # use CustomLabelBinarizer instead of LabelBinarizer(), which is supposed to work on Labels only
    ])

# Full Transformation Pipeline deal with all previous transformation
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# Put into Dataframe
housing_prepared_df = pd.DataFrame(
    housing_prepared,
    columns = num_attribs + ["rooms_per_household", "population_per_household","bedrooms_per_room"] + list(encoder.classes_) ,
    index = list(housing.index.values))
housing_prepared_df.head()



##################
# Model Training #
##################

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared_df, housing_labels)
housing_labels.iloc[:5] # iloc: integer position

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared_df)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
lin_rmse

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

# DecisionTree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared_df, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared_df)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # overfitting

# DecisionTree - Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared_df, housing_labels,
                         scoring="neg_mean_squared_error", cv=10) # 10 folds
tree_rmse_scores = np.sqrt(-scores) # cross-validation uses utility function, so use negative value

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)


# RandomForest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared_df, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores) # cross-validation uses utility function, so use negative value
display_scores(forest_rmse_scores)

# SVM - support vector machine regressor
from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared_df, housing_labels)
svm_reg_scores = cross_val_score(svm_reg, housing_prepared_df, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-forest_scores) # cross-validation uses utility function, so use negative value
display_scores(forest_rmse_scores)
# SVM using grid search
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search.fit(housing_prepared_df, housing_labels)
# Find the best hyperparameter combination
grid_search.best_params_
grid_search.best_estimator_


##############
# Tune Model #
##############

# Grid Search
from sklearn.model_selection import GridSearchCV # or RandomizedSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    # can add more models here
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
# Find the best hyperparameter combination
grid_search.best_params_
grid_search.best_estimator_
# Print each hyperparameter combination tested during the grid search
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
# Randomized Search: preferable when hyperparameter space is large
from sklearn.model_selection import RandomizedSearchCV  
from scipy.stats import randint
param_distribs = {
        'n_estimators': randint(low=1, high=200), # default size return a single int, size = (2,4), 2x4 matrix
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42) # each iteration test a random combination over 5 folds
rnd_search.fit(housing_prepared, housing_labels)    
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params) 
 

# Display importance score
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
sorted(zip(feature_importances, list(housing_prepared_df.columns)), reverse=True) # easier than the original code
 


###############################
# Evaluate model on test data #
###############################  
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1) # x
y_test = strat_test_set["median_house_value"].copy()     # label
x_incomplete_rows = X_test[X_test.isnull().any(axis=1)] # check NaNs in attributes
y_incomplete_rows = y_test[y_test.isnull()] # check Nans in labels

X_test_prepared = full_pipeline.transform(X_test)
#X_test_prepared2 = full_pipeline.fit_transform(X_test) # they perform the same, book says don't use fit_transform

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# 95% confidence interval for the test RMSE. Standard error = std/sqrt(df)
import scipy.stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)
np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))
# Manually t
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# Use Z-score since n is large
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m) # standard deviation of the sampling distribution, = std/sqrt(df)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


#########
# Extra #
#########

# Full Pipeline for Preparation + Modeling + Prediction
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])
full_pipeline_with_predictor.fit(housing_prepared_df, housing_labels)
full_pipeline_with_predictor.predict(X_test_prepared)



############
# Exercise #
############

# 1.Support Vector Machine regressor with GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search.fit(housing_prepared_df, housing_labels)

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

grid_search.best_params_
grid_search.best_estimator_


# 2.RandomizedSearchCV with SVM
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_distribs = {
        'kernel': ['linear', 'rbf'], # gamma is ignored when kernel is "linear"
        'C': reciprocal(20, 200000), # reciprocal continuous random variable, (a, b) are shape parameters
        'gamma': expon(scale=1.0), # exponential distribution with lamda = 1
    }
# The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be
# The exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be
svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
rnd_search.best_estimator_


# 3.Create a transformer in the preparation pipeline to select only the most important attributes
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared_df

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
feature_importances_sorted_ = sorted(zip(feature_importances, list(housing_prepared_df.columns)), reverse=True)

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared_df

# Select features have importance >= rating
from sklearn.base import BaseEstimator, TransformerMixin
class importance_selector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, rating):
        self.feature_importances = feature_importances
        self.rating = rating
    def fit(self, X, y=None):
        self.feature_indices_ = np.where(self.feature_importances >= self.rating) # returns an array inside array, we need the one inside
        return self
    def transform(self, X):
        return X[:, self.feature_indices_[0]] 

rating = 0.05
feature_indices_ = np.where(feature_importances >= rating)
feature_indices_ = feature_indices_[0]

preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', importance_selector(feature_importances, rating))
]) 

housing_prepared_feature_selection = preparation_and_feature_selection_pipeline.fit_transform(housing)
attributes_prepared = list(housing_prepared_df)
attributes_prepared_feature_select = attributes_prepared[list(feature_indices_)]

housing_prepared_feature_selection_df = pd.DataFrame( # transform to dataframe
        housing_prepared_feature_selection,
        columns = (attributes_prepared[indice] for indice in feature_indices_),
        index = list(housing_prepared_df.index.values))


# 4.Create a single pipeline that does the full data preparation plus the final prediction
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', importance_selector(feature_importances, rating)),
    ('svm_reg', SVR(**rnd_search.best_params_))  # svm_reg = SVR()
])

prepare_select_and_predict_pipeline.fit(housing_prepared_feature_selection_df, housing_labels)
# validation
some_data = housing_prepared_feature_selection_df.iloc[:4]
some_labels = housing_labels.iloc[:4]
print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))


# 5.Automatically explore some preparation options using GridSearchCV.
from scipy.stats import randint
housing = strat_train_set.copy() # original training data
param_grid = [
        {'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
         'feature_selection__rating': list(feature_importances)}
]
grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2, n_jobs=4) # Verbose is a general programming term for produce lots of logging output
grid_search_prep.fit(housing, housing_labels)
grid_search_prep.best_params_
grid_search_prep.best_estimator_