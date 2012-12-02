import warnings
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from collections import defaultdict

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false

from sklearn.utils.sparsefuncs import mean_variance_axis0
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import _histogram, _collect_indices,\
    _fair_array_counts, resample_labels

from sklearn import datasets
from sklearn.linear_model.stochastic_gradient import SGDClassifier

iris = datasets.load_iris()


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def test_scaler_1d():
    """Test scaling of dataset along single axis"""
    rng = np.random.RandomState(0)
    X = rng.randn(5)
    X_orig_copy = X.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

    # check inverse transform
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert_array_almost_equal(X_scaled_back, X_orig_copy)

    # Test with 1D list
    X = [0., 1., 2, 0.4, 1.]
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

    X_scaled = scale(X)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 1.0)


def test_scaler_2d_arrays():
    """Test scaling of 2d array along first axis"""
    rng = np.random.RandomState(0)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero

    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_false(np.any(np.isnan(X_scaled)))

    assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
    # Check that X has been copied
    assert_true(X_scaled is not X)

    # check inverse transform
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert_true(X_scaled_back is not X)
    assert_true(X_scaled_back is not X_scaled)
    assert_array_almost_equal(X_scaled_back, X)

    X_scaled = scale(X, axis=1, with_std=False)
    assert_false(np.any(np.isnan(X_scaled)))
    assert_array_almost_equal(X_scaled.mean(axis=1), 4 * [0.0])
    X_scaled = scale(X, axis=1, with_std=True)
    assert_false(np.any(np.isnan(X_scaled)))
    assert_array_almost_equal(X_scaled.mean(axis=1), 4 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=1), 4 * [1.0])
    # Check that the data hasn't been modified
    assert_true(X_scaled is not X)

    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert_false(np.any(np.isnan(X_scaled)))
    assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
    # Check that X has not been copied
    assert_true(X_scaled is X)

    X = rng.randn(4, 5)
    X[:, 0] = 1.0  # first feature is a constant, non zero feature
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_false(np.any(np.isnan(X_scaled)))
    assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
    # Check that X has not been copied
    assert_true(X_scaled is not X)


def test_min_max_scaler_iris():
    X = iris.data
    scaler = MinMaxScaler()
    # default params
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), 0)
    assert_array_almost_equal(X_trans.min(axis=0), 0)
    assert_array_almost_equal(X_trans.max(axis=0), 1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # not default params: min=1, max=2
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), 1)
    assert_array_almost_equal(X_trans.max(axis=0), 2)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # min=-.5, max=.6
    scaler = MinMaxScaler(feature_range=(-.5, .6))
    X_trans = scaler.fit_transform(X)
    assert_array_almost_equal(X_trans.min(axis=0), -.5)
    assert_array_almost_equal(X_trans.max(axis=0), .6)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    # raises on invalid range
    scaler = MinMaxScaler(feature_range=(2, 1))
    assert_raises(ValueError, scaler.fit, X)


def test_min_max_scaler_zero_variance_features():
    """Check min max scaler on toy data with zero variance features"""
    X = [[0.,  1.,  0.5],
         [0.,  1., -0.1],
         [0.,  1.,  1.1]]

    X_new = [[+0.,  2.,  0.5],
             [-1.,  1.,  0.0],
             [+0.,  1.,  1.5]]

    # default params
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    X_expected_0_1 = [[0.,  0.,  0.5],
                      [0.,  0.,  0.0],
                      [0.,  0.,  1.0]]
    assert_array_almost_equal(X_trans, X_expected_0_1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_almost_equal(X, X_trans_inv)

    X_trans_new = scaler.transform(X_new)
    X_expected_0_1_new = [[+0.,  1.,  0.500],
                          [-1.,  0.,  0.083],
                          [+0.,  0.,  1.333]]
    assert_array_almost_equal(X_trans_new, X_expected_0_1_new, decimal=2)

    # not default params
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    X_expected_1_2 = [[1.,  1.,  1.5],
                      [1.,  1.,  1.0],
                      [1.,  1.,  2.0]]
    assert_array_almost_equal(X_trans, X_expected_1_2)


def test_scaler_without_centering():
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)

    assert_raises(ValueError, StandardScaler().fit, X_csr)

    null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
    X_null = null_transform.fit_transform(X_csr)
    assert_array_equal(X_null.data, X_csr.data)
    X_orig = null_transform.inverse_transform(X_null)
    assert_array_equal(X_orig.data, X_csr.data)

    scaler = StandardScaler(with_mean=False).fit(X)
    X_scaled = scaler.transform(X, copy=True)
    assert_false(np.any(np.isnan(X_scaled)))

    scaler_csr = StandardScaler(with_mean=False).fit(X_csr)
    X_csr_scaled = scaler_csr.transform(X_csr, copy=True)
    assert_false(np.any(np.isnan(X_csr_scaled.data)))

    scaler_csc = StandardScaler(with_mean=False).fit(X_csc)
    X_csc_scaled = scaler_csr.transform(X_csc, copy=True)
    assert_false(np.any(np.isnan(X_csc_scaled.data)))

    assert_equal(scaler.mean_, scaler_csr.mean_)
    assert_array_almost_equal(scaler.std_, scaler_csr.std_)

    assert_equal(scaler.mean_, scaler_csc.mean_)
    assert_array_almost_equal(scaler.std_, scaler_csc.std_)

    assert_array_almost_equal(
        X_scaled.mean(axis=0), [0., -0.01,  2.24, -0.35, -0.78], 2)
    assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])

    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis0(X_csr_scaled)
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))

    # Check that X has not been modified (copy)
    assert_true(X_scaled is not X)
    assert_true(X_csr_scaled is not X_csr)

    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert_true(X_scaled_back is not X)
    assert_true(X_scaled_back is not X_scaled)
    assert_array_almost_equal(X_scaled_back, X)

    X_csr_scaled_back = scaler_csr.inverse_transform(X_csr_scaled)
    assert_true(X_csr_scaled_back is not X_csr)
    assert_true(X_csr_scaled_back is not X_csr_scaled)
    assert_array_almost_equal(X_csr_scaled_back.toarray(), X)

    X_csc_scaled_back = scaler_csr.inverse_transform(X_csc_scaled.tocsc())
    assert_true(X_csc_scaled_back is not X_csc)
    assert_true(X_csc_scaled_back is not X_csc_scaled)
    assert_array_almost_equal(X_csc_scaled_back.toarray(), X)


def test_scaler_without_copy():
    """Check that StandardScaler.fit does not change input"""
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sp.csr_matrix(X)

    X_copy = X.copy()
    StandardScaler(copy=False).fit(X)
    assert_array_equal(X, X_copy)

    X_csr_copy = X_csr.copy()
    StandardScaler(with_mean=False, copy=False).fit(X_csr)
    assert_array_equal(X_csr.toarray(), X_csr_copy.toarray())


def test_scale_sparse_with_mean_raise_exception():
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X_csr = sp.csr_matrix(X)

    # check scaling and fit with direct calls on sparse data
    assert_raises(ValueError, scale, X_csr, with_mean=True)
    assert_raises(ValueError, StandardScaler(with_mean=True).fit, X_csr)

    # check transform and inverse_transform after a fit on a dense array
    scaler = StandardScaler(with_mean=True).fit(X)
    assert_raises(ValueError, scaler.transform, X_csr)

    X_transformed_csr = sp.csr_matrix(scaler.transform(X))
    assert_raises(ValueError, scaler.inverse_transform, X_transformed_csr)


def test_scale_function_without_centering():
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0  # first feature is always of zero
    X_csr = sp.csr_matrix(X)

    X_scaled = scale(X, with_mean=False)
    assert_false(np.any(np.isnan(X_scaled)))

    X_csr_scaled = scale(X_csr, with_mean=False)
    assert_false(np.any(np.isnan(X_csr_scaled.data)))

    # test csc has same outcome
    X_csc_scaled = scale(X_csr.tocsc(), with_mean=False)
    assert_array_almost_equal(X_scaled, X_csc_scaled.toarray())

    # raises value error on axis != 0
    assert_raises(ValueError, scale, X_csr, with_mean=False, axis=1)

    assert_array_almost_equal(X_scaled.mean(axis=0),
                              [0., -0.01,  2.24, -0.35, -0.78], 2)
    assert_array_almost_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
    # Check that X has not been copied
    assert_true(X_scaled is not X)

    X_csr_scaled_mean, X_csr_scaled_std = mean_variance_axis0(X_csr_scaled)
    assert_array_almost_equal(X_csr_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_csr_scaled_std, X_scaled.std(axis=0))


def test_warning_scaling_integers():
    """Check warning when scaling integer data"""
    X = np.array([[1, 2, 0],
                  [0, 0, 0]], dtype=np.uint8)

    with warnings.catch_warnings(record=True) as w:
        StandardScaler().fit(X)
        assert_equal(len(w), 1)

    with warnings.catch_warnings(record=True) as w:
        MinMaxScaler().fit(X)
        assert_equal(len(w), 1)


def test_normalizer_l1():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = sp.csr_matrix(X_dense)

    # set the row number 3 to zero
    X_dense[3, :] = 0.0

    # set the row number 3 to zero without pruning (can happen in real life)
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # build the pruned variant using the regular constructor
    X_sparse_pruned = sp.csr_matrix(X_dense)

    # check inputs that support the no-copy optim
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):

        normalizer = Normalizer(norm='l1', copy=True)
        X_norm = normalizer.transform(X)
        assert_true(X_norm is not X)
        X_norm1 = toarray(X_norm)

        normalizer = Normalizer(norm='l1', copy=False)
        X_norm = normalizer.transform(X)
        assert_true(X_norm is X)
        X_norm2 = toarray(X_norm)

        for X_norm in (X_norm1, X_norm2):
            row_sums = np.abs(X_norm).sum(axis=1)
            for i in range(3):
                assert_almost_equal(row_sums[i], 1.0)
            assert_almost_equal(row_sums[3], 0.0)

    # check input for which copy=False won't prevent a copy
    for init in (sp.coo_matrix, sp.csc_matrix, sp.lil_matrix):
        X = init(X_dense)
        X_norm = normalizer = Normalizer(norm='l2', copy=False).transform(X)

        assert_true(X_norm is not X)
        assert_true(isinstance(X_norm, sp.csr_matrix))

        X_norm = toarray(X_norm)
        for i in range(3):
            assert_almost_equal(row_sums[i], 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_normalizer_l2():
    rng = np.random.RandomState(0)
    X_dense = rng.randn(4, 5)
    X_sparse_unpruned = sp.csr_matrix(X_dense)

    # set the row number 3 to zero
    X_dense[3, :] = 0.0

    # set the row number 3 to zero without pruning (can happen in real life)
    indptr_3 = X_sparse_unpruned.indptr[3]
    indptr_4 = X_sparse_unpruned.indptr[4]
    X_sparse_unpruned.data[indptr_3:indptr_4] = 0.0

    # build the pruned variant using the regular constructor
    X_sparse_pruned = sp.csr_matrix(X_dense)

    # check inputs that support the no-copy optim
    for X in (X_dense, X_sparse_pruned, X_sparse_unpruned):

        normalizer = Normalizer(norm='l2', copy=True)
        X_norm1 = normalizer.transform(X)
        assert_true(X_norm1 is not X)
        X_norm1 = toarray(X_norm1)

        normalizer = Normalizer(norm='l2', copy=False)
        X_norm2 = normalizer.transform(X)
        assert_true(X_norm2 is X)
        X_norm2 = toarray(X_norm2)

        for X_norm in (X_norm1, X_norm2):
            for i in range(3):
                assert_almost_equal(la.norm(X_norm[i]), 1.0)
            assert_almost_equal(la.norm(X_norm[3]), 0.0)

    # check input for which copy=False won't prevent a copy
    for init in (sp.coo_matrix, sp.csc_matrix, sp.lil_matrix):
        X = init(X_dense)
        X_norm = normalizer = Normalizer(norm='l2', copy=False).transform(X)

        assert_true(X_norm is not X)
        assert_true(isinstance(X_norm, sp.csr_matrix))

        X_norm = toarray(X_norm)
        for i in range(3):
            assert_almost_equal(la.norm(X_norm[i]), 1.0)
        assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_normalize_errors():
    """Check that invalid arguments yield ValueError"""
    assert_raises(ValueError, normalize, [[0]], axis=2)
    assert_raises(ValueError, normalize, [[0]], norm='l3')


def test_binarizer():
    X_ = np.array([[1, 0, 5], [2, 3, -1]])

    for init in (np.array, list, sp.csr_matrix, sp.csc_matrix):

        X = init(X_.copy())

        binarizer = Binarizer(threshold=2.0, copy=True)
        X_bin = toarray(binarizer.transform(X))
        assert_equal(np.sum(X_bin == 0), 4)
        assert_equal(np.sum(X_bin == 1), 2)
        X_bin = binarizer.transform(X)
        assert_equal(sp.issparse(X), sp.issparse(X_bin))

        binarizer = Binarizer(copy=True).fit(X)
        X_bin = toarray(binarizer.transform(X))
        assert_true(X_bin is not X)
        assert_equal(np.sum(X_bin == 0), 2)
        assert_equal(np.sum(X_bin == 1), 4)

        binarizer = Binarizer(copy=True)
        X_bin = binarizer.transform(X)
        assert_true(X_bin is not X)
        X_bin = toarray(X_bin)
        assert_equal(np.sum(X_bin == 0), 2)
        assert_equal(np.sum(X_bin == 1), 4)

        binarizer = Binarizer(copy=False)
        X_bin = binarizer.transform(X)
        if init is not list:
            assert_true(X_bin is X)
        X_bin = toarray(X_bin)
        assert_equal(np.sum(X_bin == 0), 2)
        assert_equal(np.sum(X_bin == 1), 4)

    binarizer = Binarizer(threshold=-0.5, copy=True)
    for init in (np.array, list):
        X = init(X_.copy())

        X_bin = toarray(binarizer.transform(X))
        assert_equal(np.sum(X_bin == 0), 1)
        assert_equal(np.sum(X_bin == 1), 5)
        X_bin = binarizer.transform(X)

    # Cannot use threshold < 0 for sparse
    assert_raises(ValueError, binarizer.transform, sp.csc_matrix(X))


def test_label_binarizer():
    lb = LabelBinarizer()

    # two-class case
    inp = ["neg", "pos", "pos", "neg"]
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # multi-class case
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = np.array([[0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [1, 0, 0, 0]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_set_label_encoding():
    lb = LabelBinarizer(neg_label=-2, pos_label=2)

    # two-class case
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[-2, 2, 2, -2]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # multi-class case
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array([[-2, -2, -2, +2],
                         [-2, -2, +2, -2],
                         [-2, +2, -2, -2],
                         [-2, -2, +2, -2],
                         [+2, -2, -2, -2]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_multilabel():
    lb = LabelBinarizer()

    # test input as lists of tuples
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1],
                              [1, 0, 0],
                              [1, 1, 0]])
    got = lb.fit_transform(inp)
    assert_array_equal(indicator_mat, got)
    assert_equal(lb.inverse_transform(got), inp)

    # test input as label indicator matrix
    lb.fit(indicator_mat)
    assert_array_equal(indicator_mat,
                       lb.inverse_transform(indicator_mat))

    # regression test for the two-class multilabel case
    lb = LabelBinarizer()

    inp = [[1, 0], [0], [1], [0, 1]]
    expected = np.array([[1, 1],
                         [1, 0],
                         [0, 1],
                         [1, 1]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_equal([set(x) for x in lb.inverse_transform(got)],
                 [set(x) for x in inp])


def test_label_binarizer_errors():
    """Check that invalid arguments yield ValueError"""
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)

    multi_label = [(2, 3), (0,), (0, 2)]
    assert_raises(ValueError, lb.transform, multi_label)

    lb = LabelBinarizer()
    assert_raises(ValueError, lb.transform, [])
    assert_raises(ValueError, lb.inverse_transform, [])

    assert_raises(ValueError, LabelBinarizer, neg_label=2, pos_label=1)
    assert_raises(ValueError, LabelBinarizer, neg_label=2, pos_label=2)


def test_one_hot_encoder():
    """Test OneHotEncoder's fit and transform."""
    X = [[3, 2, 1], [0, 1, 1]]
    enc = OneHotEncoder()
    # discover max values automatically
    X_trans = enc.fit_transform(X).toarray()
    assert_equal(X_trans.shape, (2, 5))
    assert_array_equal(enc.active_features_,
                       np.where([1, 0, 0, 1, 0, 1, 1, 0, 1])[0])
    assert_array_equal(enc.feature_indices_, [0, 4, 7, 9])

    # check outcome
    assert_array_equal(X_trans,
                       [[0., 1., 0., 1., 1.],
                        [1., 0., 1., 0., 1.]])

    # max value given as 3
    enc = OneHotEncoder(n_values=4)
    X_trans = enc.fit_transform(X)
    assert_equal(X_trans.shape, (2, 4 * 3))
    assert_array_equal(enc.feature_indices_, [0, 4, 8, 12])

    # max value given per feature
    enc = OneHotEncoder(n_values=[3, 2, 2])
    X = [[1, 0, 1], [0, 1, 1]]
    X_trans = enc.fit_transform(X)
    assert_equal(X_trans.shape, (2, 3 + 2 + 2))
    assert_array_equal(enc.n_values_, [3, 2, 2])
    # check that testing with larger feature works:
    X = np.array([[2, 0, 1], [0, 1, 1]])
    enc.transform(X)

    # test that an error is raise when out of bounds:
    X_too_large = [[0, 2, 1], [0, 1, 1]]
    assert_raises(ValueError, enc.transform, X_too_large)

    # test that error is raised when wrong number of features
    assert_raises(ValueError, enc.transform, X[:, :-1])
    # test that error is raised when wrong number of features in fit
    # with prespecified n_values
    assert_raises(ValueError, enc.fit, X[:, :-1])
    # test exception on wrong init param
    assert_raises(TypeError, OneHotEncoder(n_values=np.int).fit, X)

    enc = OneHotEncoder()
    # test negative input to fit
    assert_raises(ValueError, enc.fit, [[0], [-1]])

    # test negative input to transform
    enc.fit([[0], [1]])
    assert_raises(ValueError, enc.transform, [[0], [-1]])


def test_label_encoder():
    """Test LabelEncoder's transform and inverse_transform methods"""
    le = LabelEncoder()
    le.fit([1, 1, 4, 5, -1, 0])
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    assert_array_equal(le.transform([0, 1, 4, 4, 5, -1, -1]),
                       [1, 2, 3, 3, 4, 0, 0])
    assert_array_equal(le.inverse_transform([1, 2, 3, 3, 4, 0, 0]),
                       [0, 1, 4, 4, 5, -1, -1])
    assert_raises(ValueError, le.transform, [0, 6])


def test_label_encoder_fit_transform():
    """Test fit_transform"""
    le = LabelEncoder()
    ret = le.fit_transform([1, 1, 4, 5, -1, 0])
    assert_array_equal(ret, [2, 2, 3, 4, 0, 1])

    le = LabelEncoder()
    ret = le.fit_transform(["paris", "paris", "tokyo", "amsterdam"])
    assert_array_equal(ret, [1, 1, 2, 0])


def test_label_encoder_string_labels():
    """Test LabelEncoder's transform and inverse_transform methods with
    non-numeric labels"""
    le = LabelEncoder()
    le.fit(["paris", "paris", "tokyo", "amsterdam"])
    assert_array_equal(le.classes_, ["amsterdam", "paris", "tokyo"])
    assert_array_equal(le.transform(["tokyo", "tokyo", "paris"]),
                       [2, 2, 1])
    assert_array_equal(le.inverse_transform([2, 2, 1]),
                       ["tokyo", "tokyo", "paris"])
    assert_raises(ValueError, le.transform, ["london"])


def test_label_encoder_errors():
    """Check that invalid arguments yield ValueError"""
    le = LabelEncoder()
    assert_raises(ValueError, le.transform, [])
    assert_raises(ValueError, le.inverse_transform, [])


def test_label_binarizer_iris():
    lb = LabelBinarizer()
    Y = lb.fit_transform(iris.target)
    clfs = [SGDClassifier().fit(iris.data, Y[:, k])
            for k in range(len(lb.classes_))]
    Y_pred = np.array([clf.decision_function(iris.data) for clf in clfs]).T
    y_pred = lb.inverse_transform(Y_pred)
    accuracy = np.mean(iris.target == y_pred)
    y_pred2 = SGDClassifier().fit(iris.data, iris.target).predict(iris.data)
    accuracy2 = np.mean(iris.target == y_pred2)
    assert_almost_equal(accuracy, accuracy2)


def test_label_binarizer_multilabel_unlabeled():
    """Check that LabelBinarizer can handle an unlabeled sample"""
    lb = LabelBinarizer()
    y = [[1, 2], [1], []]
    Y = np.array([[1, 1],
                  [1, 0],
                  [0, 0]])
    assert_array_equal(lb.fit_transform(y), Y)


def test_center_kernel():
    """Test that KernelCenterer is equivalent to StandardScaler
       in feature space"""
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    scaler = StandardScaler(with_std=False)
    scaler.fit(X_fit)
    X_fit_centered = scaler.transform(X_fit)
    K_fit = np.dot(X_fit, X_fit.T)

    # center fit time matrix
    centerer = KernelCenterer()
    K_fit_centered = np.dot(X_fit_centered, X_fit_centered.T)
    K_fit_centered2 = centerer.fit_transform(K_fit)
    assert_array_almost_equal(K_fit_centered, K_fit_centered2)

    # center predict time matrix
    X_pred = rng.random_sample((2, 4))
    K_pred = np.dot(X_pred, X_fit.T)
    X_pred_centered = scaler.transform(X_pred)
    K_pred_centered = np.dot(X_pred_centered, X_fit_centered.T)
    K_pred_centered2 = centerer.transform(K_pred)
    assert_array_almost_equal(K_pred_centered, K_pred_centered2)


def test_fit_transform():
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    for obj in ((StandardScaler(), Normalizer(), Binarizer())):
        X_transformed = obj.fit(X).transform(X)
        X_transformed2 = obj.fit_transform(X)
        assert_array_equal(X_transformed, X_transformed2)


def test_add_dummy_feature():
    X = [[1, 0], [0, 1], [0, 1]]
    X = add_dummy_feature(X)
    assert_array_equal(X, [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_coo():
    X = sp.coo_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert_true(sp.isspmatrix_coo(X), X)
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_csc():
    X = sp.csc_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert_true(sp.isspmatrix_csc(X), X)
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


def test_add_dummy_feature_csr():
    X = sp.csr_matrix([[1, 0], [0, 1], [0, 1]])
    X = add_dummy_feature(X)
    assert_true(sp.isspmatrix_csr(X), X)
    assert_array_equal(X.toarray(), [[1, 1, 0], [1, 0, 1], [1, 0, 1]])


y = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])


def test_histogram():
    assert_equal(_histogram(y), {1: 1, 2: 2, 3: 3, 4: 4})


def test_collect_indices():
    result = _collect_indices(y)

    expected = {1: np.array([0]),
                2: np.array([1, 2]),
                3: np.array([3, 4, 5]),
                4: np.array([6, 7, 8, 9])}
    assert_equal(result.keys(), expected.keys())
    for key in result:
        assert_array_equal(result[key], expected[key])


def test_resample_labels_same_distribution():
    indices = resample_labels(y)
    assert_equal(len(indices), 10)
    assert_array_equal(y, y[np.sort(indices)])

    indices = resample_labels(y, replace=False)
    assert_equal(len(indices), 10)
    assert_array_equal(y, y[np.sort(indices)])

    indices = resample_labels(y, replace=True)
    assert_equal(len(indices), 10)

    indices = resample_labels(y, scaling=2.0, replace=False)
    assert_array_equal(y[np.sort(indices)], np.repeat(y, 2))

    indices = resample_labels(y, scaling=20, replace=False)
    assert_array_equal(y[np.sort(indices)], np.repeat(y, 2))

    indices = resample_labels(y, scaling=2.0, replace=False, shuffle=True)
    assert_array_equal(y[np.sort(indices)], np.sort(np.repeat(y, 2)))

    indices = resample_labels(y, scaling=20, replace=False, shuffle=True)
    assert_array_equal(y[np.sort(indices)], np.sort(np.repeat(y, 2)))

    indices = resample_labels(y, scaling=2.0, replace=True)
    assert_equal(len(indices), 20)

    indices = resample_labels(y, scaling=20, replace=True)
    assert_equal(len(indices), 20)


def test_resample_balanced():
    indices = resample_labels(y, scaling=2.0, distribution="balance",
                              replace=False)
    assert_equal(len(indices), 2 * len(y))

    indices = resample_labels(y, scaling=20, distribution="balance",
                              replace=False)
    assert_equal(len(indices), 2 * len(y))

    indices = resample_labels(y, scaling=2.0, distribution="balance",
                              replace=True)
    assert_equal(len(indices), 2 * len(y))

    indices = resample_labels(y, scaling=20, distribution="balance",
                              replace=True)
    assert_equal(len(indices), 2 * len(y))

    # [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
    z = np.concatenate([[i] * (4) for i in range(101)])
    check_mean_std(z, 50.0, 29.1547)

    # Should be equal because we sample everything and don't replace
    indices = resample_labels(z, distribution=None, replace=False,
                              random_state=42)
    assert_array_equal(z, z[indices])

    # Sample with replacement, should get nearly same mean/std as y
    assert_almost_equal(np.mean(z), 50.0, 3)
    indices = resample_labels(z, distribution=None, replace=True,
                              random_state=42)
    assert_almost_equal(np.mean(z[indices]), 51.2425, 3)
    assert_almost_equal(np.std(z[indices]), 28.5119, 3)

    indices = resample_labels(z, distribution=None, scaling=20000,
                              replace=True, random_state=42)
    assert_almost_equal(np.mean(z[indices]), 50.0067, 3)
    assert_almost_equal(np.std(z[indices]), 29.2240, 3)


def test_resample_labels_oversample():
    indices = resample_labels(y, distribution="oversample", replace=False)
    assert_equal(len(indices), 16)

    indices = resample_labels(y, distribution="oversample", replace=True)
    assert_equal(len(indices), 16)


def test_resample_labels_undersample():
    indices = resample_labels(y, distribution="undersample", replace=False)
    assert_equal(len(indices), 4)

    indices = resample_labels(y, distribution="undersample", replace=True)
    assert_equal(len(indices), 4)


def test_resample_labels_dict():
    indices = resample_labels(y, scaling=2.0,
                              distribution={1: .3, 2: .1, 3: .5, 4: .1},
                              random_state=42)
    assert_equal(len(indices), 2 * len(y))
    assert_equal(_histogram(y[indices]), {1: 8, 2: 2, 3: 8, 4: 2})

    indices = resample_labels(y, scaling=100,
                              distribution={1: .3, 2: .1, 3: .5, 4: .1},
                              random_state=42)
    assert_equal(len(indices), 100)
    assert_equal(_histogram(y[indices]), {1: 34, 2: 12, 3: 45, 4: 9})

    indices = resample_labels(y, scaling=100,
                              distribution={1: .3, 2: .7, 999: 0},
                              random_state=42)
    assert_equal(len(indices), 100)
    assert_equal(_histogram(y[indices]), {1: 34, 2: 66})

    indices = resample_labels(y, scaling=2.0,
                              distribution={1: .3, 2: .1, 3: .5, 4: .1},
                              replace=True, random_state=42)
    assert_equal(len(indices), 2 * len(y))
    assert_equal(_histogram(y[indices]), {1: 8, 2: 2, 3: 8, 4: 2})

    indices = resample_labels(y, scaling=100,
                              distribution={1: .3, 2: .1, 3: .5, 4: .1},
                              replace=True, random_state=42)
    assert_equal(len(indices), 100)
    assert_equal(_histogram(y[indices]), {1: 34, 2: 12, 3: 45, 4: 9})

    indices = resample_labels(y, scaling=100,
                              distribution={1: .3, 2: .7, 999: 0},
                              replace=True, random_state=42)
    assert_equal(len(indices), 100)
    assert_equal(_histogram(y[indices]), {1: 34, 2: 66})

    indices = resample_labels(y, scaling=100000,
                              distribution={1: .3, 2: .7, 999: 0},
                              replace=True,
                              random_state=42)
    assert_equal(len(indices), 100000)
    assert_equal(_histogram(y[indices]), {1: 30055, 2: 69945})


def test_resample_labels_invalid_parameters():
    assert_raises(ValueError, resample_labels, y, scaling=2.0,
                  distribution="badstring")
    assert_raises(ValueError, resample_labels, y, distribution="badstring")
    assert_raises(ValueError, resample_labels, y, distribution={1: .5})
    assert_raises(ValueError, resample_labels, y, distribution={})
    assert_raises(ValueError, resample_labels, y, distribution=555)
    assert_raises(ValueError, _fair_array_counts, 2, 5)


def check_mean_std(y, expected_mean, expected_std):
    indices = resample_labels(y, distribution="balance", replace=False,
                              random_state=42)
    assert_almost_equal(np.mean(y[indices]), expected_mean, 3)
    assert_almost_equal(np.std(y[indices]), expected_std, 3)

    indices = resample_labels(y, distribution="balance", replace=True,
                              random_state=42)
    assert_almost_equal(np.mean(y[indices]), expected_mean, 3)
    assert_almost_equal(np.std(y[indices]), expected_std, 3)

    # Scale the dataset and check the invariant
    indices = resample_labels(y, distribution="balance",
                              scaling=4.0, replace=False, random_state=42)
    assert_almost_equal(np.mean(y[indices]), expected_mean, 3)
    assert_almost_equal(np.std(y[indices]), expected_std, 3)

    indices = resample_labels(y, distribution="balance", scaling=4.0,
                              replace=True, random_state=42)
    assert_almost_equal(np.mean(y[indices]), expected_mean, 3)
    assert_almost_equal(np.std(y[indices]), expected_std, 3)


def test_resample_labels_linearly_unbalanced():
    # [0, 1,1, 2,2,2, ...]
    y = np.concatenate([[i] * (i + 1) for i in range(101)])
    check_mean_std(y, 50.0, 29.1547)

    # Should be equal because we sample everything and don't replace
    indices = resample_labels(y, distribution=None, replace=False,
                              random_state=42)
    assert_array_equal(y, y[indices])

    # Sample with replacement, should get nearly same mean/std as y
    assert_almost_equal(np.mean(y), 66.6666, 3)
    indices = resample_labels(y, distribution=None, replace=True,
                              random_state=42)
    assert_almost_equal(np.mean(y[indices]), 67.1574, 3)
    assert_almost_equal(np.std(y[indices]), 23.5837, 3)

    indices = resample_labels(y, distribution=None, scaling=10000,
                              replace=True, random_state=42)
    assert_almost_equal(np.mean(y[indices]), 67.0377, 3)
    assert_almost_equal(np.std(y[indices]), 23.6514, 3)


def test_resample_labels_quadratically_unbalanced():
    # [0, 1,1, 2,2,2,2, ...]
    y = np.concatenate([[i] * 2 ** i for i in range(11)])
    # Multiple of 11 for exact mean
    y = y[:2002]
    check_mean_std(y, 5.0, 3.1622)

    # Should be equal because we sample everything and don't replace
    indices = resample_labels(y, distribution=None, replace=False,
                              random_state=42)
    assert_array_equal(y, y[indices])

    # Sample with replacement, should get nearly same mean/std as y
    assert_almost_equal(np.mean(y), 8.9830, 3)
    indices = resample_labels(y, distribution=None, replace=True,
                              random_state=42)
    assert_almost_equal(np.mean(y[indices]), 8.9755, 3)
    assert_almost_equal(np.std(y[indices]), 1.4740, 3)

    indices = resample_labels(y, scaling=4004, distribution=None, replace=True,
                              random_state=42)
    assert_almost_equal(np.mean(y[indices]), 8.9822, 3)
    assert_almost_equal(np.std(y[indices]), 1.4315, 3)


def test_resample_labels_shuffled():
    assert_array_equal(
        resample_labels(y, replace=False, shuffle=False, random_state=42),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    assert_array_equal(
        resample_labels(y, replace=False, shuffle=True, random_state=42),
        np.array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6]))

    assert_array_equal(
        resample_labels(y, replace=True, shuffle=False, random_state=42),
        np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4]))

    assert_array_equal(
        resample_labels(y, replace=True, shuffle=True, random_state=42),
        np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4]))


def test_resample_labels_distribution_shuffled():
    assert_array_equal(
        resample_labels(y, distribution="balance", replace=False,
                        shuffle=False,
                        random_state=42),
        np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    )

    assert_array_equal(
        resample_labels(y, distribution="balance", replace=True,
                        shuffle=False,
                        random_state=42),
        np.array([0, 0, 0, 1, 2, 3, 3, 5, 7, 8])
    )

    assert_array_equal(
        resample_labels(y, distribution="balance", replace=False,
                        shuffle=True,
                        random_state=42),
        np.array([7, 5, 0, 1, 3, 6, 0, 0, 4, 2])
    )

    assert_array_equal(
        resample_labels(y, distribution="balance", replace=True,
                        shuffle=True,
                        random_state=42),
        np.array([0, 3, 8, 0, 7, 0, 1, 2, 5, 3])
    )


def test_resample_labels_dict_shuffled():
    assert_array_equal(
        resample_labels(y, distribution={1: .5, 2: .5},
                        replace=False, shuffle=False,
                        random_state=42),
        np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    )

    assert_array_equal(
        resample_labels(y, distribution={1: .5, 2: .5},
                        replace=False, shuffle=True,
                        random_state=42),
        np.array([2, 0, 0, 1, 2, 2, 0, 0, 1, 1])
    )

    assert_array_equal(
        resample_labels(y, distribution={1: .5, 2: .5},
                        replace=True, shuffle=False,
                        random_state=42),
        np.array([0, 0, 0, 0, 2, 1, 2, 2, 2, 2])
    )

    assert_array_equal(
        resample_labels(y, distribution={1: .5, 2: .5},
                        replace=True, shuffle=True,
                        random_state=42),
        np.array([2, 0, 2, 2, 0, 2, 0, 2, 0, 1])
    )
