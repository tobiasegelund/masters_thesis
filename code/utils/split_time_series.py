from typing import TypeVar, Dict
DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')


class SplitOnYears():
    """
    A class that split data on years. All previous data prior to the year will be used as train
    (expanding window). If block_split is True it will not use expanding window but sliding window
    as in reference.

    :param kfolds_years: A dictionary representing the folds
    """
    def __init__(self, kfolds_years: Dict):
        self.n_splits = len(kfolds_years.keys())
        self.kfolds_years = kfolds_years
        self.train_indices = dict()
        self.test_indices = dict()

    def get_indices(self, X: DataFrame, varname: str):
        _kfolds_years = self.kfolds_years

        indices_controller = list()
        for fold, years in _kfolds_years.items():
            idx = X[X[varname].isin(years)].index.tolist()
            indices_controller += idx
            self.train_indices[fold] = indices_controller.copy()
            self.test_indices[fold] = idx

    def split(self, block_split=False):
        for fold in range(self.n_splits):
            if fold == (self.n_splits - 1):
                break

            if block_split:
                yield self.test_indices[fold], self.test_indices[fold + 1]
            else:
                yield self.train_indices[fold], self.test_indices[fold + 1]


class SplitOnTime():
    """
    A class that split data on time. All previous data prior to the year will be used as train
    (expanding window). If block_split is True it will not use expanding window but sliding window
    as in reference.

    :param n_splits: The number of CV splits
    """
    def __init__(self, n_splits: int):
        self.n_splits = n_splits
        self.train_indices = dict()
        self.test_indices = dict()

    def get_indices(self, X: DataFrame, varname: str):
        fold_size = X[varname].max() // self.n_splits
        for fold in range(self.n_splits):
            start = fold * fold_size
            stop = start + fold_size
            idx_test = X[(X[varname] > start) & (X[varname] <= stop)].index.tolist()
            if fold == (self.n_splits - 1):  # Use the rest as test data
                idx_train = X[(X[varname] <= X[varname].max())].index.tolist()
            else:
                idx_train = X[(X[varname] <= stop)].index.tolist()
            self.train_indices[fold] = idx_train
            self.test_indices[fold] = idx_test

    def split(self, block_split=False):
        for fold in range(self.n_splits):
            if fold == (self.n_splits - 1):
                break

            if block_split:
                yield self.test_indices[fold], self.test_indices[fold + 1]
            else:
                yield self.train_indices[fold], self.test_indices[fold + 1]


class WalkForwardSplit():
    """
    A class that split data into Walk Foward Split

    :param train_size: The level of training size (first fold)
    """
    def __init__(self, train_size: float):
        self.train_size = train_size
        self.train_indices = dict()
        self.train_indices_block = dict()
        self.test_indices = dict()
        self.test_size = None

    def get_indices(self, X: DataFrame, varname: str):
        _train_size = int(X[varname].max() * self.train_size)
        _test_size = X[varname].max() - _train_size
        self.test_size = _test_size

        for fold in range(self.test_size + 1):
            start = fold
            end = _train_size + fold

            idx_train = X[(X[varname] <= end)].index.tolist()
            self.train_indices[fold] = idx_train

            idx_train_block = X[(X[varname] > start) & (X[varname] <= end)].index.tolist()
            self.train_indices_block[fold] = idx_train_block

            idx_test = X[(X[varname] == end)].index.tolist()
            self.test_indices[fold] = idx_test

    def split(self):
        for fold in range(self.test_size):
            yield self.train_indices[fold], self.test_indices[fold + 1]
