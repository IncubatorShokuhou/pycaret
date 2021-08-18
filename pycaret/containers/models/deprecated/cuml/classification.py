# Module: containers.models.classification
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

import logging
from typing import Dict

# The purpose of this module is to serve as a central repository of classification models. The `classification` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.
import cuml
import numpy as np

import pycaret.containers.base_container
import pycaret.internal.cuml_wrappers
from pycaret.containers.models.deprecated.base_model import (
    leftover_parameters_to_categorical_distributions,
)
from pycaret.containers.models.deprecated.classification import (
    AdaBoostClassifierContainer,
    BaggingClassifierContainer,
    CalibratedClassifierCVContainer,
    CatBoostClassifierContainer,
    ClassifierContainer,
    DecisionTreeClassifierContainer,
    ExtraTreesClassifierContainer,
    GaussianNBClassifierContainer,
    GaussianProcessClassifierContainer,
    GradientBoostingClassifierContainer,
    KNeighborsClassifierContainer,
    LGBMClassifierContainer,
    LinearDiscriminantAnalysisContainer,
    LogisticRegressionClassifierContainer,
    MLPClassifierContainer,
    QuadraticDiscriminantAnalysisContainer,
    RandomForestClassifierContainer,
    RidgeClassifierContainer,
    SGDClassifierContainer,
    StackingClassifierContainer,
    SVCClassifierContainer,
    VotingClassifierContainer,
    XGBClassifierContainer,
)
from pycaret.internal.cuml_wrappers import get_ridge_classifier, get_svc_classifier
from pycaret.internal.distributions import IntUniformDistribution, UniformDistribution
from pycaret.internal.utils import get_class_name, get_logger, np_list_arange


class CumlLogisticRegressionClassifierContainer(LogisticRegressionClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        np.random.seed(globals_dict["seed"])
        from cuml.linear_model import LogisticRegression

        args = {"max_iter": 1000}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["C"] = np_list_arange(0.001, 10, 0.001, inclusive=True)

        tune_grid["penalty"] = ["l2", "l1"]

        tune_distributions["C"] = UniformDistribution(0.001, 10)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=LogisticRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=True,
        )


class CumlKNeighborsClassifierContainer(KNeighborsClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from cuml.neighbors import KNeighborsClassifier

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["n_neighbors"] = range(1, 51)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 51)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=KNeighborsClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=True,
        )


class CumlGaussianNBClassifierContainer(GaussianNBClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from cuml.naive_bayes import GaussianNB  # new in 21.10!

        # TODO: should we change the search grid?
        args = {}
        tune_args = {}
        tune_grid = {
            "var_smoothing": [
                0.000000001,
                0.000000002,
                0.000000005,
                0.000000008,
                0.000000009,
                0.0000001,
                0.0000002,
                0.0000003,
                0.0000005,
                0.0000007,
                0.0000009,
                0.00001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.007,
                0.009,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.1,
                1,
            ]
        }
        tune_distributions = {
            "var_smoothing": UniformDistribution(0.000000001, 1, log=True)
        }

        super().__init__(
            class_def=GaussianNB,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=True,
        )


class CumlDecisionTreeClassifierContainer(DecisionTreeClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlSGDClassifierContainer(SGDClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        from cuml import MBSGDClassifier as SGDClassifier

        args = {"tol": 0.001, "loss": "hinge", "penalty": "l2", "eta0": 0.001}
        tune_args = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.0002,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.0000000001, 0.9999999999),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "eta0": UniformDistribution(0.001, 0.5, log=True),
        }

        batch_size = [
            (512, 50000),
            (256, 25000),
            (128, 10000),
            (64, 5000),
            (32, 1000),
            (16, 0),
        ]
        for arg, x_len in batch_size:
            if len(globals_dict["X_train"]) >= x_len:
                args["batch_size"] = arg
                break

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=SGDClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=True,
        )


class CumlSVCClassifierContainer(SVCClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        SVC = get_svc_classifier()

        args = {
            "gamma": "auto",
            "C": 1.0,
            "probability": True,
            "kernel": "rbf",
            "random_state": globals_dict["seed"],
        }
        tune_args = {}
        tune_grid = {
            "C": np_list_arange(0, 50, 0.01, inclusive=True),
            "class_weight": ["balanced", {}],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=SVC,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
            is_gpu_enabled=True,
        )


class CumlGaussianProcessClassifierContainer(GaussianProcessClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlMLPClassifierContainer(MLPClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlRidgeClassifierContainer(RidgeClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        RidgeClassifier = get_ridge_classifier()

        tune_grid = dict(normalize=[True, False])
        tune_grid["alpha"] = np_list_arange(0.01, 10, 0.01, inclusive=False)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = UniformDistribution(0.001, 10)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=RidgeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=True,
        )
        self.reference = get_class_name(cuml.linear_model.Ridge)


class CumlRandomForestClassifierContainer(RandomForestClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        RandomForestClassifier = (
            pycaret.internal.cuml_wrappers.get_random_forest_classifier()
        )

        args = {"seed": globals_dict["seed"]}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.0002,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }

        tune_grid["split_criterion"] = [0, 1]

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=RandomForestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=True,
        )
        self.reference = get_class_name(cuml.ensemble.RandomForestClassifier)


class CumlQuadraticDiscriminantAnalysisContainer(
    QuadraticDiscriminantAnalysisContainer
):
    def __init__(self):
        super().__init__()


class CumlAdaBoostClassifierContainer(AdaBoostClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlGradientBoostingClassifierContainer(GradientBoostingClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlLinearDiscriminantAnalysisContainer(LinearDiscriminantAnalysisContainer):
    def __init__(self):
        super().__init__()


class CumlExtraTreesClassifierContainer(ExtraTreesClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlXGBClassifierContainer(XGBClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
            "verbosity": 0,
            "booster": "gbtree",
            "tree_method": "gpu_hist",
        }

        super().__init__(
            args=args,
            is_gpu_enabled=True,
        )


class CumlLGBMClassifierContainer(LGBMClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from lightgbm import LGBMClassifier
        from lightgbm.basic import LightGBMError

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }

        try:
            lgb = LGBMClassifier(device="gpu")
            lgb.fit(np.zeros((2, 2)), [0, 1])
            is_gpu_enabled = "gpu"
            del lgb
            args["device"] = "gpu"
        except:
            try:
                lgb = LGBMClassifier(device="cuda")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                is_gpu_enabled = "cuda"
                del lgb
                args["device"] = "cuda"
            except LightGBMError:
                is_gpu_enabled = False
                if globals_dict["gpu_param"] == "force":
                    raise RuntimeError(
                        f"LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                    )

        super().__init__(
            args=args,
            is_gpu_enabled=True,
        )


class CumlCatBoostClassifierContainer(CatBoostClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        try:
            import catboost
        except ImportError:
            logger.warning("Couldn't import catboost.CatBoostClassifier")
            self.active = False
            return

        catboost_version = tuple([int(x) for x in catboost.__version__.split(".")])
        if catboost_version < (0, 23, 2):
            logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost_version}"
            )
            self.active = False
            return

        from catboost import CatBoostClassifier

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        use_gpu = (
            globals_dict["gpu_n_jobs_param"] and len(globals_dict["X_train"]) >= 50000
        )

        args = {
            "random_state": globals_dict["seed"],
            "verbose": False,
            "thread_count": globals_dict["n_jobs_param"],
            "task_type": "GPU" if use_gpu else "CPU",
            "border_count": 32 if use_gpu else 254,
        }
        tune_args = {}
        tune_grid = {
            "eta": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "depth": list(range(1, 12)),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "random_strength": np_list_arange(0, 0.8, 0.1, inclusive=True),
            "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200],
        }
        tune_distributions = {
            "eta": UniformDistribution(0.000001, 0.5, log=True),
            "depth": IntUniformDistribution(1, 11),
            "n_estimators": IntUniformDistribution(10, 300),
            "random_strength": UniformDistribution(0, 0.8),
            "l2_leaf_reg": IntUniformDistribution(1, 200, log=True),
        }

        if use_gpu:
            tune_grid["depth"] = list(range(1, 9))
            tune_distributions["depth"] = (IntUniformDistribution(1, 8),)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            class_def=CatBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=use_gpu,
        )


class CumlBaggingClassifierContainer(BaggingClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlStackingClassifierContainer(StackingClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlVotingClassifierContainer(VotingClassifierContainer):
    def __init__(self):
        super().__init__()


class CumlCalibratedClassifierCVContainer(CalibratedClassifierCVContainer):
    def __init__(self):
        super().__init__()


def get_all_model_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, ClassifierContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, ClassifierContainer, raise_errors
    )
