from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from dask_ml.ensemble import BlockwiseVotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pycaret.internal.tunable import TunableMLPClassifier

try:
    class DaskKNeighborsClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = KNeighborsClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskKNeighborsClassifier = None


def get_dask_kneighbors_classifier():
    return DaskKNeighborsClassifier

try:
    class DaskDecisionTreeClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = DecisionTreeClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskDecisionTreeClassifier = None


def get_dask_decision_tree_classifier():
    return DaskDecisionTreeClassifier

try:
    class DaskSGDClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = SGDClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskSGDClassifier = None


def get_dask_sgd_classifier():
    return DaskSGDClassifier


try:
    class DaskSVC(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = SVC(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskSVC = None


def get_dask_svc_classifier():
    return DaskSVC



try:
    class DaskGaussianProcessClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = GaussianProcessClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskGaussianProcessClassifier = None


def get_dask_gussian_process_classifier():
    return DaskGaussianProcessClassifier


try:
    class DaskMLPClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = MLPClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskMLPClassifier = None


def get_dask_mlp_classifier():
    return DaskMLPClassifier

try:
    class DaskTunableMLPClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = TunableMLPClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskTunableMLPClassifier = None


def get_dask_tunable_mlp_classifier():
    return DaskTunableMLPClassifier


try:
    class DaskRidgeClassifier(BlockwiseVotingClassifier):
        def __init__(self, **kwargs):
            self.estimator = RidgeClassifier(**kwargs)
            super().__init__(estimator=self.estimator, classes=[])

except ImportError:
    DaskRidgeClassifier = None


def get_dask_ridge_classifier():
    return DaskRidgeClassifier