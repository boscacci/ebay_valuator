import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-5.57580127035496
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        SelectFwe(score_func=f_regression, alpha=0.01)
    ),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=20, n_estimators=100, nthread=1, subsample=0.45)),
    RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=3, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
