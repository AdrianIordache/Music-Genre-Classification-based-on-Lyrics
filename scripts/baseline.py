import numpy as np
import pandas as pd
from IPython.display import display

import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF


from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RANSACRegressor, PassiveAggressiveRegressor, LassoLars, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, AdaBoostRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OrdinalEncoder

import warnings
warnings.filterwarnings("ignore")

SEED = 42
train_df = pd.read_csv("../data/Lyrics-Genre-Train.csv") 
test_df  = pd.read_csv("../data/Lyrics-Genre-Test-GroundTruth.csv")

X_train = train_df['Lyrics'].values
X_test  = test_df['Lyrics'].values

y_train   = train_df['Genre'].values.reshape(-1, 1)
y_test    = test_df['Genre'].values.reshape(-1, 1)

encoder = OrdinalEncoder()
y_train = encoder.fit_transform(y_train)
y_test  = encoder.transform(y_test)

estimators = [
	("lr", LogisticRegression(n_jobs = -1)),
	("svr", LinearSVC(random_state = SEED)),
	("svr_2", SVC(C = 0.1)),
	("rf", RandomForestClassifier(n_estimators = 30, random_state = SEED, n_jobs = -1)),
	("extra", ExtraTreesClassifier(n_estimators = 30, random_state = SEED, n_jobs = -1)),
	("lgbm", LGBMClassifier(n_estimators = 30, n_jobs = -1, random_state = SEED)),
]


tfv = TfidfVectorizer(ngram_range = (1, 7), analyzer = 'char_wb')
tfv.fit(X_train)

X_train = tfv.transform(X_train)
X_test = tfv.transform(X_test)

print(X_train.shape)

model = StackingClassifier(estimators = estimators, final_estimator = XGBClassifier(n_jobs = -1, random_state = SEED, learning_rate = 0.7, n_estimators = 50), n_jobs = 1)
model.fit(X_train, y_train)

y_predict_test = model.predict(X_test)

print("Bseline Accuracy: {}".format(accuracy_score(y_test, y_predict_test)))
