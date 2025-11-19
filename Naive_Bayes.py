import math
import numpy as np
import pandas as pd

class Naive_Bayes:
    def __init__(self, train_df):
        """
        train_df: pandas.DataFrame where last column is the label
        features: all columns except the last
        """
        self.train_df = train_df.reset_index(drop=True)
        if isinstance(train_df, pd.DataFrame):
            self.feature_cols = train_df.columns[:-1].tolist()
            self.label_col = train_df.columns[-1]
        else:
            raise ValueError("train_df must be a pandas DataFrame")
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.vars = {}
        self.fit()

    def fit(self):
        df = self.train_df
        self.classes = df[self.label_col].unique().tolist()
        total = len(df)
        for cls in self.classes:
            subset = df[df[self.label_col] == cls][self.feature_cols]
            self.class_priors[cls] = len(subset) / total
            self.means[cls] = subset.mean().to_dict()
            # use population variance (ddof=0); avoid zero variance
            self.vars[cls] = subset.var(ddof=0).replace(0, 1e-6).to_dict()

    def _gaussian_log_prob(self, x, mean, var):
        # return log of gaussian pdf
        coeff = -0.5 * math.log(2 * math.pi * var)
        exp_term = -((x - mean) ** 2) / (2 * var)
        return coeff + exp_term

    def predict_row(self, row):
        # row: pandas Series or sequence with feature columns accessible by name
        best_cls = None
        best_log_prob = -math.inf
        for cls in self.classes:
            log_prob = math.log(self.class_priors.get(cls, 1e-9))
            for feat in self.feature_cols:
                x = row[feat]
                mean = self.means[cls][feat]
                var = self.vars[cls][feat]
                log_prob += self._gaussian_log_prob(x, mean, var)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_cls = cls
        return best_cls

    def predict(self, df):
        return df.apply(lambda r: self.predict_row(r), axis=1)

    def test(self, test_df):
        test_df = test_df.reset_index(drop=True)
        y_true = test_df[self.label_col]
        y_pred = self.predict(test_df)
        accuracy = (y_true == y_pred).mean()
        cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion matrix:")
        print(cm)