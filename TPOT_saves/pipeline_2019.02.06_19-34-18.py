import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-5.656306822179316
exported_pipeline = XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=8, n_estimators=100, nthread=1, subsample=0.7000000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
