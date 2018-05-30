from __future__ import division
import numpy as np
from scipy import sparse
from sklearn.model_selection import LeaveOneOut

from sklearn.utils.testing import (assert_array_almost_equal, assert_equal,
                                   assert_greater,
                                   assert_array_equal,
                                   assert_raises,
                                   ignore_warnings)
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import brier_score_loss
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputClassifier

from civismlext.multioutput import MultiOutputCalibratedClassifierCV


def make_multioutput_classification(n_dvs=2, **kwargs):
    X, y = make_classification(**kwargs)
    ys = [shuffle(y, random_state=i) for i in range(n_dvs-1)]
    # prepend the original DV
    ys = [y] + ys
    Y = np.vstack(tuple(ys)).T
    return X, Y


@ignore_warnings
def test_multioutput_calibration():
    """Test calibration objects with isotonic and sigmoid"""
    n_dvs = 2
    n_samples = 100
    X, Y = make_multioutput_classification(n_dvs=n_dvs,
                                           n_samples=2 * n_samples,
                                           n_features=6, random_state=42)

    sample_weight = np.random.RandomState(seed=42).uniform(size=X.shape[0])

    X -= X.min()  # MultinomialNB only allows positive X

    # split train and test
    X_train, Y_train, sw_train = \
        X[:n_samples], Y[:n_samples], sample_weight[:n_samples]
    X_test, Y_test = X[n_samples:], Y[n_samples:]

    # MultiOutput Naive-Bayes
    clf = MultiOutputClassifier(MultinomialNB()).fit(X_train, Y_train,
                                                     sample_weight=sw_train)
    proba = clf.predict_proba(X_test)
    prob_pos_clf = proba[0][:, 1]
    assert len(proba) == n_dvs, (
        "Expected %d arrays (one per DV), got %d" % (n_dvs, len(proba)))

    pc_clf = MultiOutputCalibratedClassifierCV(clf, cv=Y.shape[0] + 1)
    assert_raises(ValueError, pc_clf.fit, X, Y)

    # MultiOutput Naive Bayes with calibration
    for this_X_train, this_X_test in [(X_train, X_test),
                                      (sparse.csr_matrix(X_train),
                                       sparse.csr_matrix(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            pc_clf = MultiOutputCalibratedClassifierCV(
                clf, method=method, cv=2)
            # Note that this fit overwrites the fit on the entire training
            # set
            pc_clf.fit(this_X_train, Y_train, sample_weight=sw_train)
            prob_pos_pc_clf = pc_clf.predict_proba(this_X_test)[0][:, 1]

            # Check that brier score has improved after calibration
            # Only check the first DV, since the others are shuffled
            assert_greater(brier_score_loss(Y_test[:, 0], prob_pos_clf),
                           brier_score_loss(Y_test[:, 0], prob_pos_pc_clf)),

            # Check invariance against relabeling [0, 1] -> [1, 2]
            pc_clf.fit(this_X_train, Y_train + 1, sample_weight=sw_train)
            prob_pos_pc_clf_relabeled = pc_clf.predict_proba(
                this_X_test)[0][:, 1]
            assert_array_almost_equal(prob_pos_pc_clf,
                                      prob_pos_pc_clf_relabeled)

            # Check invariance against relabeling [0, 1] -> [-1, 1]
            pc_clf.fit(this_X_train, 2 * Y_train - 1, sample_weight=sw_train)
            prob_pos_pc_clf_relabeled = pc_clf.predict_proba(
                this_X_test)[0][:, 1]
            assert_array_almost_equal(prob_pos_pc_clf,
                                      prob_pos_pc_clf_relabeled)

            # Check invariance against relabeling [0, 1] -> [1, 0]
            pc_clf.fit(this_X_train, (Y_train + 1) % 2,
                       sample_weight=sw_train)
            prob_pos_pc_clf_relabeled = \
                pc_clf.predict_proba(this_X_test)[0][:, 1]
            if method == "sigmoid":
                # TODO: multioutput seems to have caused this to be
                # *not quite* invariant to the default 6 places. Why?
                assert_array_almost_equal(prob_pos_pc_clf,
                                          1 - prob_pos_pc_clf_relabeled,
                                          decimal=5)
            else:
                # Isotonic calibration is not invariant against relabeling
                # but should improve in both cases
                assert_greater(brier_score_loss(Y_test[:, 0], prob_pos_clf),
                               brier_score_loss((Y_test[:, 0] + 1) % 2,
                                                prob_pos_pc_clf_relabeled))

        # Check failure cases:
        # only "isotonic" and "sigmoid" should be accepted as methods
        clf_invalid_method = MultiOutputCalibratedClassifierCV(
            clf, method="foo")
        assert_raises(ValueError, clf_invalid_method.fit, X_train, Y_train)

        # base-estimators should provide either decision_function or
        # predict_proba (most regressors, for instance, should fail)
        clf_base_regressor = \
            MultiOutputCalibratedClassifierCV(
                RandomForestRegressor(), method="sigmoid")
        assert_raises(RuntimeError, clf_base_regressor.fit, X_train, Y_train)


def test_sample_weight():
    n_samples = 100
    X, Y = make_multioutput_classification(n_samples=2 * n_samples,
                                           n_features=6, random_state=42)

    sample_weight = np.random.RandomState(seed=42).uniform(size=Y.shape[0])
    X_train, Y_train, sw_train = \
        X[:n_samples], Y[:n_samples], sample_weight[:n_samples]
    X_test = X[n_samples:]

    for method in ['sigmoid', 'isotonic']:
        base_estimator = RandomForestClassifier(random_state=42)
        calibrated_clf = MultiOutputCalibratedClassifierCV(base_estimator,
                                                           method=method)
        calibrated_clf.fit(X_train, Y_train, sample_weight=sw_train)
        probs_with_sw = calibrated_clf.predict_proba(X_test)

        # As the weights are used for the calibration, they should still yield
        # a different predictions
        calibrated_clf.fit(X_train, Y_train)
        probs_without_sw = calibrated_clf.predict_proba(X_test)

        diff = np.linalg.norm(probs_with_sw[0] - probs_without_sw[0])
        assert_greater(diff, 0.1)


# TODO: implement decision_function option, and fix test
# def test_multioutput_calibration_multiclass():
#     """Test calibration for multiclass """
#     # test multi-output multi-class setting with classifier that implements
#     # only decision function
#     clf = ClassifierChain(LinearSVC())
#     X, y_idx = make_blobs(n_samples=100, n_features=2, random_state=42,
#                           centers=3, cluster_std=3.0)

#     # Use categorical labels to check that CalibratedClassifierCV supports
#     # them correctly
#     target_names = np.array(['a', 'b', 'c'])
#     y_1 = target_names[y_idx]
#     y_2 = shuffle(y_1, random_state=111)
#     Y = np.vstack((y_1, y_2)).T

#     X_train, Y_train = X[::2], Y[::2]
#     X_test, Y_test = X[1::2], Y[1::2]

#     clf.fit(X_train, Y_train)
#     for method in ['isotonic', 'sigmoid']:
#         cal_clf = MultiOutputCalibratedClassifierCV(clf, method=method, cv=2)
#         cal_clf.fit(X_train, Y_train)
#         probas = cal_clf.predict_proba(X_test)
#         assert_array_almost_equal(
#             np.sum(probas[0], axis=1), np.ones(len(X_test)))

#         # Check that log-loss of calibrated classifier is smaller than
#         # log-loss of naively turned OvR decision function to probabilities
#         # via softmax
#         def softmax(y_pred):
#             e = np.exp(-y_pred)
#             return e / e.sum(axis=1).reshape(-1, 1)

#         uncalibrated_log_loss = \
#             log_loss(Y_test[:,0], softmax(clf.decision_function(X_test)))
#         calibrated_log_loss = log_loss(Y_test, probas[0])
#         assert_greater_equal(uncalibrated_log_loss, calibrated_log_loss)

#     # Test that calibration of a multiclass classifier decreases log-loss
#     # for RandomForestClassifier
#     X, y = make_blobs(n_samples=100, n_features=2, random_state=42,
#                       cluster_std=3.0)
#     X_train, y_train = X[::2], y[::2]
#     X_test, y_test = X[1::2], y[1::2]

#     clf = RandomForestClassifier(n_estimators=10, random_state=42)
#     clf.fit(X_train, y_train)
#     clf_probs = clf.predict_proba(X_test)
#     loss = log_loss(y_test, clf_probs)

#     for method in ['isotonic', 'sigmoid']:
#         cal_clf = CalibratedClassifierCV(clf, method=method, cv=3)
#         cal_clf.fit(X_train, y_train)
#         cal_clf_probs = cal_clf.predict_proba(X_test)
#         cal_loss = log_loss(y_test, cal_clf_probs)
#         assert_greater(loss, cal_loss)


def test_calibration_prefit():
    """Test calibration for prefitted classifiers"""
    n_samples = 50
    n_dvs = 3
    X, Y = make_multioutput_classification(n_dvs,
                                           n_samples=3 * n_samples,
                                           n_features=6,
                                           random_state=42)
    sample_weight = np.random.RandomState(seed=42).uniform(size=Y.shape[0])

    X -= X.min()  # MultinomialNB only allows positive X

    # split train and test
    X_train, Y_train, sw_train = \
        X[:n_samples], Y[:n_samples], sample_weight[:n_samples]
    X_calib, Y_calib, sw_calib = \
        X[n_samples:2 * n_samples], Y[n_samples:2 * n_samples], \
        sample_weight[n_samples:2 * n_samples]
    X_test, Y_test = X[2 * n_samples:], Y[2 * n_samples:]

    clf = MultiOutputClassifier(MultinomialNB())
    clf.fit(X_train, Y_train, sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[0][:, 1]

    # Naive Bayes with calibration
    for this_X_calib, this_X_test in [(X_calib, X_test),
                                      (sparse.csr_matrix(X_calib),
                                       sparse.csr_matrix(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            pc_clf = MultiOutputCalibratedClassifierCV(
                clf, method=method, cv="prefit")

            for sw in [sw_calib, None]:
                pc_clf.fit(this_X_calib, Y_calib, sample_weight=sw)
                y_prob = pc_clf.predict_proba(this_X_test)[0]
                y_pred = pc_clf.predict(this_X_test)[:, 0]
                prob_pos_pc_clf = y_prob[:, 1]
                assert_array_equal(y_pred,
                                   np.array([0, 1])[np.argmax(y_prob, axis=1)])

                assert_greater(brier_score_loss(Y_test[:, 0], prob_pos_clf),
                               brier_score_loss(Y_test[:, 0], prob_pos_pc_clf))


def test_multioutput_calibration_nan_imputer():
    """Test that calibration can accept nan"""
    n_dvs = 3
    X, Y = make_multioutput_classification(n_dvs=n_dvs, n_samples=10,
                                           n_features=2,
                                           n_informative=2, n_redundant=0,
                                           random_state=42)
    X[0, 0] = np.nan
    clf = Pipeline(
        [('imputer', Imputer()),
         ('rf', RandomForestClassifier(n_estimators=1))])
    clf_c = MultiOutputCalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_c.fit(X, Y)
    clf_c.predict(X)


def test_multioutput_calibration_prob_sum():
    # Test that sum of probabilities is 1. A non-regression test for
    # issue #7796
    num_classes = 2
    n_dvs = 2
    X, Y = make_multioutput_classification(n_dvs=n_dvs, n_samples=10,
                                           n_features=5,
                                           n_classes=num_classes)
    clf = RandomForestClassifier()
    clf_prob = MultiOutputCalibratedClassifierCV(clf, method="sigmoid",
                                                 cv=LeaveOneOut())
    clf_prob.fit(X, Y)

    probs = clf_prob.predict_proba(X)[0]
    assert_array_almost_equal(probs.sum(axis=1), np.ones(probs.shape[0]))


def test_multioutput_calibration_less_classes():
    # Test to check calibration works fine when train set in a test-train
    # split does not contain all classes
    # Since this test uses LOO, at each iteration train set will not contain a
    # class label
    X = np.random.randn(10, 5)
    X -= X.min()  # MultinomialNB only allows positive X
    y = np.arange(10)
    Y = np.vstack((y, shuffle(y, random_state=999))).T

    clf = MultiOutputClassifier(MultinomialNB())
    cal_clf = MultiOutputCalibratedClassifierCV(clf, method="sigmoid",
                                                cv=LeaveOneOut())
    cal_clf.fit(X, Y)

    for i, calibrated_classifier in enumerate(cal_clf.calibrated_classifiers_):
        proba = calibrated_classifier.predict_proba(X)[0]
        assert_array_equal(proba[:, i], np.zeros(len(y)))
        assert_equal(np.all(np.hstack([proba[:, :i],
                                       proba[:, i + 1:]])), True)
