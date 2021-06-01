# k-folds
K_FOLDS = {
    0: [2007, 2008, 2009, 2010],
    1: [2011, 2012],
    2: [2013, 2014],
    3: [2015, 2016],
    4: [2017, 2018]
}

K_FOLDS_TRAIN = {
    0: [2007, 2008, 2009, 2010],
    1: [2007, 2008, 2009, 2010, 2011, 2012],
    2: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
    3: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
}

K_FOLDS_TEST = {
    0: [2007, 2008, 2009, 2010],
    1: [2011, 2012],
    2: [2013, 2014],
    3: [2015, 2016],
    4: [2017, 2018]
}

# block folds
BLOCK_FOLDS = {
    0: [2007, 2008, 2009, 2010],
    1: [2011],
    2: [2012],
    3: [2013],
    4: [2014],
    5: [2015],
    6: [2016],
    7: [2017],
    8: [2018]
}

BLOCK_FOLDS_TRAIN = {
    0: [2007, 2008, 2009, 2010],
    1: [2007, 2008, 2009, 2010, 2011],
    2: [2007, 2008, 2009, 2010, 2011, 2012],
    3: [2007, 2008, 2009, 2010, 2011, 2012, 2013],
    4: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
    5: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    6: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016],
    7: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
}

BLOCK_FOLDS_TEST = {
    0: [2011],
    1: [2012],
    2: [2013],
    3: [2014],
    4: [2015],
    5: [2016],
    6: [2017],
    7: [2018]
}

# LightGBM hyperparameters
MIN_DATA_IN_LEAFS = [50, 250, 500]
MAX_DEPTHS = [5, 15, 30]
NUM_LEAVESS = [10, 50, 80, 120]
ESTIMATORS = [500, 1000, 2000, 3000]
LEARNING_RATES = [0.01, 0.001, 0.0001]

# Correlation thresholds
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# optimal hyperparameters lgbm
PARAMS_PRICE = {
    "boosting_type": 'gbdt',
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    'metric': 'rmse',
    'num_leaves': 80,
    'min_child_samples': 250
}

PARAMS_SALES = {
    "boosting_type": 'gbdt',
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    'metric': 'rmse',
    'num_leaves': 80,
    'min_child_samples': 50
}

# nn preprocessing configs
CATEGORICAL_FEATURES = [
    'period_sin',
    'period_cos',
    'Month',
    'Year',
    'opbevaring',
    'tilskud',
    'number_of_bidders',
    'Substitution Group Name'
]

NOT_INCLUDE_FEATURES = [
    't+1',
    't',
    't-1',
    't-2',
    't-3',
    't-4',
    't-5',
    't-6',
    't-7',
    't-12',
    't-25',
    't+1 sales',
    't sales',
    't-1 sales',
    't-2 sales',
    't-3 sales',
    't-4 sales',
    't-5 sales',
    't-6 sales',
    't-7 sales',
    't-12 sales',
    't-25 sales'
]
