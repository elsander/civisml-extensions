from __future__ import division
import warnings

import numpy as np

from sklearn.preprocessing import LabelEncoder, label_binarize, LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_X_y, check_array, indexable
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from sklearn.utils.fixes import signature
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import check_cv
from sklearn.calibration import _SigmoidCalibration


class MultiOutputCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration for multioutput classifiers with isotonic
    regression or sigmoid. Each dependent variable is calibrated independently.
    Based on `sklearn.calibration.CalibratedClassifierCV`.

    With this class, the base_estimator is fit on the train set of the
    cross-validation generator and the test set is used for calibration.
    The probabilities for each of the folds are then averaged
    for prediction. In case that cv="prefit" is passed to __init__,
    it is assumed that base_estimator has been fitted already and all
    data is used for calibration. Note that data for fitting the
    classifier and for calibrating it must be disjoint.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv=prefit, the
        classifier must have been fit already on data.

    method : 'sigmoid' or 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach. It is not advised to use isotonic calibration
        with too few calibration samples ``(<<1000)`` since it tends to
        overfit.
        Use sigmoids (Platt's calibration) in this case.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

    Attributes
    ----------
    classes_ : list(n_dvs) array, shape (n_classes)
        The class labels for each dependent variable.

    calibrated_classifiers_ : list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each crossvalidation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

   References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """

    def __init__(self, base_estimator=None, method='sigmoid', cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_dvs)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        n_dvs = y.shape[1]
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False, multi_output=True)
        X, y = indexable(X, y)
        les = [LabelBinarizer().fit(y[:, i]) for i in range(n_dvs)]
        self.classes_ = [le.classes_ for le in les]

        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        for i in range(n_dvs):
            if n_folds and \
                    np.any([np.sum(y == class_) < n_folds for class_ in
                            self.classes_[i]]):
                raise ValueError("Requesting %d-fold cross-validation but "
                                 "provided less than %d examples for at least "
                                 "one class." % (n_folds, n_folds))

        self.calibrated_classifiers_ = []
        if self.base_estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            base_estimator = LinearSVC(random_state=0)
        else:
            base_estimator = self.base_estimator

        if self.cv == "prefit":
            calibrated_classifier = _MultiOutputCalibratedClassifier(
                base_estimator, method=self.method)
            if sample_weight is not None:
                calibrated_classifier.fit(X, y, sample_weight)
            else:
                calibrated_classifier.fit(X, y)
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            cv = check_cv(self.cv, y, classifier=True)
            fit_parameters = signature(base_estimator.fit).parameters
            estimator_name = type(base_estimator).__name__
            if (sample_weight is not None
                    and "sample_weight" not in fit_parameters):
                warnings.warn("%s does not support sample_weight. Samples"
                              " weights are only used for the calibration"
                              " itself." % estimator_name)
                base_estimator_sample_weight = None
            else:
                if sample_weight is not None:
                    sample_weight = check_array(sample_weight, ensure_2d=False)
                    check_consistent_length(y, sample_weight)
                base_estimator_sample_weight = sample_weight
            for train, test in cv.split(X, y):
                this_estimator = clone(base_estimator)
                if base_estimator_sample_weight is not None:
                    this_estimator.fit(
                        X[train], y[train],
                        sample_weight=base_estimator_sample_weight[train])
                else:
                    this_estimator.fit(X[train], y[train])

                calibrated_classifier = _MultiOutputCalibratedClassifier(
                    this_estimator, method=self.method,
                    classes=self.classes_)
                if sample_weight is not None:
                    calibrated_classifier.fit(X[test], y[test],
                                              sample_weight[test])
                else:
                    calibrated_classifier.fit(X[test], y[test])
                self.calibrated_classifiers_.append(calibrated_classifier)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        check_is_fitted(self, ["classes_", "calibrated_classifiers_"])
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
                        force_all_finite=False)
        # Compute the arithmetic mean of the predictions of the calibrated
        # classifiers
        mean_proba = None
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            if mean_proba is None:
                mean_proba = proba
            else:
                for i in range(len(mean_proba)):
                    mean_proba[i] += proba[i]

        for i in range(len(mean_proba)):
            mean_proba[i] /= len(self.calibrated_classifiers_)

        return mean_proba

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ["classes_", "calibrated_classifiers_"])
        proba = self.predict_proba(X)
        pred = np.zeros([X.shape[0], len(proba)])
        for i in range(len(proba)):
            pred[:, i] = self.classes_[i][np.argmax(
                proba[i], axis=1)]
        return pred


class _MultiOutputCalibratedClassifier(object):
    """Multioutput probability calibration with isotonic regression or sigmoid.

    It assumes that base_estimator has already been fit, and trains the
    calibration on the input set of the fit function. Note that this class
    should not be used as an estimator directly. Use
    MultiOutputCalibratedClassifierCV instead.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. No default value since
        it has to be an already fitted estimator.

    method : 'sigmoid' | 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.

    classes : array-like, shape (n_classes,), optional
            Contains unique classes used to fit the base estimator.
            if None, then classes is extracted from the given target values
            in fit().

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """

    def __init__(self, base_estimator, method='sigmoid', classes=None):
        self.base_estimator = base_estimator
        self.method = method
        self.classes = classes

    def _preproc(self, X, i):
        n_classes = len(self.classes_[i])
        # if hasattr(self.base_estimator, "decision_function"):
        #     df = self.base_estimator.decision_function(X)[i]
        #     if df.ndim == 1:
        #         df = df[:, np.newaxis]
        # elif hasattr(self.base_estimator, "predict_proba"):
        if hasattr(self.base_estimator, "predict_proba"):
            df = self.base_estimator.predict_proba(X)[i]
            if n_classes == 2:
                df = df[:, 1:]
        else:
            raise RuntimeError('classifier has no predict_proba method.')

        try:
            cls = self.base_estimator.classes_[i]
        except AttributeError:
            # MultiOutputClassifier doesn't have a "classes_" attribute,
            # so we can special-case by introspecting the estimator list
            cls = self.base_estimator.estimators_[i].classes_
        idx_pos_class = self.label_encoder_[i].\
            transform(cls)

        return df, idx_pos_class

    def _fit_one(self, X, y, i, sample_weight=None):
        """Calibrate the fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_dvs)
            Target values.

        i : integer
            Which column of y to fit

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        Y = label_binarize(y[:, i], self.classes_[i])

        df, idx_pos_class = self._preproc(X, i)

        calibrators = []
        for k, this_df in zip(idx_pos_class, df.T):
            if self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            elif self.method == 'sigmoid':
                calibrator = _SigmoidCalibration()
            else:
                raise ValueError('method should be "sigmoid" or '
                                 '"isotonic". Got %s.' % self.method)
            calibrator.fit(this_df, Y[:, k], sample_weight)
            calibrators.append(calibrator)
        return calibrators

    def fit(self, X, y, sample_weight=None):
        n_dvs = y.shape[1]
        self.label_encoder_ = [LabelEncoder() for i in range(n_dvs)]
        if self.classes is None:
            [self.label_encoder_[i].fit(y[:, i]) for i in range(n_dvs)]
        else:
            [self.label_encoder_[i].fit(self.classes[i]) for i in range(n_dvs)]
        self.classes_ = [self.label_encoder_[i].classes_ for i in range(n_dvs)]
        self.calibrators_ = []
        for i in range(n_dvs):
            calibrators = self._fit_one(X, y, i,
                                        sample_weight=sample_weight)
            self.calibrators_.append(calibrators)
        return self

    def predict_proba_one(self, X, i):
        n_classes = len(self.classes_[i])
        proba = np.zeros((X.shape[0], n_classes))

        df, idx_pos_class = self._preproc(X, i)

        # only get calibrators for this DV
        calibrator_list = self.calibrators_[i]

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, calibrator_list):
            if n_classes == 2:
                k += 1
            proba[:, k] = calibrator.predict(this_df)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """
        probas = []
        for i in range(len(self.classes_)):
            probas.append(self.predict_proba_one(X, i))
        return probas
