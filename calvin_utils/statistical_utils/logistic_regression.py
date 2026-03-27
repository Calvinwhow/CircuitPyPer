import numpy as np
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut

class LogisticRegression():
    """
    This is a class which will run either a multinomial logistic or a logistic regression 
    depending upon the number of things you are trying to classify. 
    
    Params:
        outcome_matrix : pd.DataFrame
            - this is the dataframe with one-hot encoding. 
            - Observations are rows, classes are columns. 
            - One "1" per row
        design_matrix : pd.DataFrame
            - This is a design matrix for typical regression setups.
            
            
    Notes:
    fitting methods
    Newton-Raphson (newton): This is the default solver. It uses the score and Hessian functions and can be very fast for models that converge well. However, it might run into problems with models that are hard to optimize.

    Limited-memory Broyden Fletcher Goldfarb Shanno Algorithm (lbfgs): Default
        This is a good general-purpose solver that works well for a wide range of problems. 
        It's especially useful when dealing with large datasets or models because it uses an approximation to the 
        Hessian to guide its search without needing to store the entire Hessian matrix.

    Broyden Fletcher Goldfarb Shanno Algorithm (bfgs): 
        Similar to lbfgs, but it uses the full Hessian matrix. 
        It can be more accurate but is less memory efficient. 
        It's more suitable for smaller datasets or models.

    Conjugate Gradient (cg): 
        This solver is useful for problems with a large number of parameters. 
        It uses an iterative method to converge to the minimum, which can be more memory-efficient for large models.

    Newton-Conjugate Gradient (ncg): 
        This method uses a line search along conjugate directions, 
        which can be more efficient for some problems compared to the standard Newton-Raphson method.

    Powell s Method (powell): 
        This is a direction-set method that does not require the calculation of gradients, 
        which can be advantageous for certain types of problems, particularly those with discontinuous derivatives.
    """
    
    def __init__(self, outcome_matrix, design_matrix):
        # outcome_matrix is expected to be ordinal labels (0/1 or 0..K)
        self.outcome_matrix = outcome_matrix
        self.design_matrix = design_matrix

    @staticmethod
    def _coerce_outcome(outcome_matrix):
        """
        Accepts ordinal labels as Series/array or a single-column DataFrame.
        If a one-hot DataFrame is passed, it is converted to ordinal via argmax.
        Returns (y, n_classes).
        """
        if isinstance(outcome_matrix, pd.DataFrame):
            if outcome_matrix.shape[1] == 1:
                y = outcome_matrix.iloc[:, 0].to_numpy()
            else:
                # Fallback: convert one-hot to ordinal labels
                y = outcome_matrix.values.argmax(1)
        else:
            y = np.asarray(outcome_matrix)
        # ensure 1-D
        y = y.reshape(-1)
        n_classes = len(np.unique(y))
        return y, n_classes
        
    def choose_method(self):
        y, n_classes = self._coerce_outcome(self.outcome_matrix)
        if n_classes > 2:
            self.multinomial_logistic()
        else:
            self.binomial_logistic()
    
    def multinomial_logistic(self):
        # Fit the regression model
        y, _ = self._coerce_outcome(self.outcome_matrix)
        model = sm.MNLogit(y, self.design_matrix)
        self.results = model.fit()
        print("----INTERPRETATION KEY----")
        cats = np.sort(np.unique(y))
        for i, cat in enumerate(cats):
            if i == 0:
                print(f"reference_category : {cat}")
            else:
                print(f"y={i-1} : {cat}")
        
    def binomial_logistic(self): 
        y, _ = self._coerce_outcome(self.outcome_matrix)
        model = sm.Logit(y, self.design_matrix)
        self.results = model.fit()
    
    @staticmethod
    def run_loocv(outcome_matrix, design_matrix):
        loo = LeaveOneOut()
        y_true = []
        y_all, n_classes = LogisticRegression._coerce_outcome(outcome_matrix)
        test_prob = np.zeros((len(y_all), max(n_classes, 2)))
        y_pred = []
        
        is_multiclass = n_classes > 2
        
        for train_index, test_index in tqdm(loo.split(design_matrix)):
            X_train, X_test = design_matrix.iloc[train_index], design_matrix.iloc[test_index]
            y_train, y_test = y_all[train_index], y_all[test_index]
            
            if is_multiclass:
                model = sm.MNLogit(y_train, X_train)
            else:
                model = sm.Logit(y_train, X_train)
            
            results = model.fit(disp=0)
            
            if is_multiclass:
                test_prob[test_index, :n_classes] = results.predict(X_test).to_numpy()
                test_pred = test_prob[test_index, :].argmax(1)
                y_true.append(y_test)
            else:
                test_prob[test_index, 1] = results.predict(X_test).to_numpy()
                test_prob[test_index, 0] = 1 - test_prob[test_index, 1]
                test_pred = (test_prob[test_index, 1] > 0.5).astype(int) # arbitrary threshold of 0.5.
                y_true.append(y_test) 
            y_pred.append(test_pred)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        if isinstance(outcome_matrix, pd.DataFrame) and outcome_matrix.shape[1] > 1:
            colnames = list(outcome_matrix.columns)
        else:
            colnames = [f"class_{i}" for i in range(test_prob.shape[1])]
        return y_true, y_pred, pd.DataFrame(test_prob, columns=colnames)

    def run(self):
        self.choose_method()
        print(self.results.summary2())
        return self.results
