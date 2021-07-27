# Module: containers.models.classification
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of classification models. The `classification` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
import pycaret.internal.cuml_wrappers
from typing import Union, Any
from pycaret.containers.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.utils import (
    param_grid_to_lists,
    get_logger,
    get_class_name,
    np_list_arange,
)
from pycaret.internal.distributions import *
import pycaret.containers.models.base_container
import numpy as np
from packaging import version


class ClassifierContainer(ModelContainer):
    """
    Base classification model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict, default = {}
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {}
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {}
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {}
        The arguments to always pass to the tuner.
    shap : bool or str, default = False
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.
    is_boosting_supported : bool, default = None
        If None, will try to automatically determine.
    is_soft_voting_supported : bool, default = None
        If None, will try to automatically determine.
    tunable : type, default = None
        If a special tunable model is used for tuning, type of
        that model, else None.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict
        The arguments to always pass to the tuner.
    shap : bool or str
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
    is_gpu_enabled : bool
        If None, will try to automatically determine.
    is_boosting_supported : bool
        If None, will try to automatically determine.
    is_soft_voting_supported : bool
        If None, will try to automatically determine.
    tunable : type
        If a special tunable model is used for tuning, type of
        that model, else None.
    """

    def __init__(
            self,
            id: str,
            name: str,
            class_def: type,
            is_turbo: bool = True,
            eq_function: Optional[type] = None,
            args: Dict[str, Any] = None,
            is_special: bool = False,
            tune_grid: Dict[str, list] = None,
            tune_distribution: Dict[str, Distribution] = None,
            tune_args: Dict[str, Any] = None,
            shap: Union[bool, str] = False,
            is_gpu_enabled: Optional[bool] = None,
            is_boosting_supported: Optional[bool] = None,
            is_soft_voting_supported: Optional[bool] = None,
            tunable: Optional[type] = None,
    ) -> None:

        self.shap = shap
        if not (isinstance(shap, bool) or shap in ["type1", "type2"]):
            raise ValueError("shap must be either bool or 'type1', 'type2'.")

        if not args:
            args = {}

        if not tune_grid:
            tune_grid = {}

        if not tune_distribution:
            tune_distribution = {}

        if not tune_args:
            tune_args = {}

        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            eq_function=eq_function,
            args=args,
            is_special=is_special,
        )
        self.is_turbo = is_turbo
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args
        self.tunable = tunable

        try:
            model_instance = class_def()

            self.is_boosting_supported = bool(
                hasattr(model_instance, "class_weights")
                or hasattr(model_instance, "predict_proba")
            )

            self.is_soft_voting_supported = bool(
                hasattr(model_instance, "predict_proba")
            )

            del model_instance
        except:
            self.is_boosting_supported = False
            self.is_soft_voting_supported = False
        finally:
            if is_boosting_supported is not None:
                self.is_boosting_supported = is_boosting_supported
            if is_soft_voting_supported is not None:
                self.is_soft_voting_supported = is_soft_voting_supported

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
            ("Turbo", self.is_turbo),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("SHAP", self.shap),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Boosting Supported", self.is_boosting_supported),
                ("Soft Voting", self.is_soft_voting_supported),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)


class LogisticRegressionClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True

        from cuml.linear_model import LogisticRegression
        logger.info("Imported cuml.linear_model.LogisticRegression")

        args = {"max_iter": 1000}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["C"] = np_list_arange(0.001, 10, 0.001, inclusive=True)

        tune_grid["penalty"] = ["l2", "l1"]
        # else:
        #     args["random_state"] = globals_dict["seed"]
        #
        #     tune_grid["class_weight"] = ["balanced", {}]

        tune_distributions["C"] = UniformDistribution(0.001, 10)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lr",
            name="Logistic Regression",
            class_def=LogisticRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class KNeighborsClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True

        from cuml.neighbors import KNeighborsClassifier

        logger.info("Imported cuml.neighbors.KNeighborsClassifier")

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
            id="knn",
            name="K Neighbors Classifier",
            class_def=KNeighborsClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


GaussianNBClassifierContainer = pycaret.containers.models.local.classification.GaussianNBClassifierContainer

DecisionTreeClassifierContainer = pycaret.containers.models.local.classification.DecisionTreeClassifierContainer

class SGDClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True
        from cuml import MBSGDClassifier as SGDClassifier
        logger.info("Imported cuml.MBSGDClassifier")

        args = {"tol": 0.001, "loss": "hinge", "penalty": "l2", "eta0": 0.001}
        tune_args = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
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
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive", "optimal"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.0000000001, 0.9999999999),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "eta0": UniformDistribution(0.001, 0.5, log=True),
        }

        tune_grid["learning_rate"].remove("optimal")
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
            id="svm",
            name="SVM - Linear Kernel",
            class_def=SGDClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class SVCClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True
        from cuml.svm import SVC
        logger.info("Imported cuml.svm.SVC")

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
            id="rbfsvm",
            name="SVM - Radial Kernel",
            class_def=SVC,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
        )

GaussianProcessClassifierContainer = pycaret.containers.models.local.classification.GaussianProcessClassifierContainer

MLPClassifierContainer = pycaret.containers.models.local.classification.MLPClassifierContainer

class RidgeClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True
        import cuml.linear_model
        logger.info("Imported cuml.linear_model")

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        RidgeClassifier = pycaret.internal.cuml_wrappers.get_ridge_classifier()

        tune_grid = {
            "normalize": [True, False],
        }

        tune_grid["alpha"] = np_list_arange(0.01, 10, 0.01, inclusive=False)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = UniformDistribution(0.001, 10)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=True,
        )
        if gpu_imported:
            self.reference = get_class_name(cuml.linear_model.Ridge)


class RandomForestClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        gpu_imported = True
        from cuml.ensemble import RandomForestClassifier
        logger.info("Imported cuml.ensemble")

        args = {
            "random_state": globals_dict["seed"],
        }

        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
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
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=True,
        )
        self.reference = get_class_name(cuml.ensemble.RandomForestClassifier)

QuadraticDiscriminantAnalysisContainer = pycaret.containers.models.local.classification.QuadraticDiscriminantAnalysisContainer

AdaBoostClassifierContainer = pycaret.containers.models.local.classification.AdaBoostClassifierContainer

GradientBoostingClassifierContainer = pycaret.containers.models.local.classification.GradientBoostingClassifierContainer

LinearDiscriminantAnalysisContainer = pycaret.containers.models.local.classification.LinearDiscriminantAnalysisContainer

ExtraTreesClassifierContainer = pycaret.containers.models.local.classification.ExtraTreesClassifierContainer

class XGBClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        try:
            import xgboost
        except ImportError:
            logger.warning("Couldn't import xgboost.XGBClassifier")
            self.active = False
            return

        if version.parse(xgboost.__version__) < version.parse("1.1.0"):
            logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost.__version__}"
            )
            self.active = False
            return

        from xgboost import XGBClassifier

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
            "verbosity": 0,
            "booster": "gbtree",
            "tree_method": "gpu_hist",
        }
        tune_args = {}
        tune_grid = {
            "learning_rate": [
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
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
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
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
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
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "subsample": UniformDistribution(0.2, 1),
            "max_depth": IntUniformDistribution(1, 11),
            "colsample_bytree": UniformDistribution(0.5, 1),
            "min_child_weight": IntUniformDistribution(1, 4),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "scale_pos_weight": UniformDistribution(1, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=True)


class LGBMClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from lightgbm import LGBMClassifier
        from lightgbm.basic import LightGBMError

        args = {
            "random_state": globals_dict["seed"],
            "n_jobs": globals_dict["n_jobs_param"],
        }
        tune_args = {}
        tune_grid = {
            "num_leaves": [
                2,
                4,
                6,
                8,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                256,
            ],
            "learning_rate": [
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
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [
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
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
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
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "feature_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_freq": [0, 1, 2, 3, 4, 5, 6, 7],
            "min_child_samples": np_list_arange(1, 100, 5, inclusive=True),
        }
        tune_distributions = {
            "num_leaves": IntUniformDistribution(2, 256),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "min_split_gain": UniformDistribution(0, 1),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "feature_fraction": UniformDistribution(0.4, 1),
            "bagging_fraction": UniformDistribution(0.4, 1),
            "bagging_freq": IntUniformDistribution(0, 7),
            "min_child_samples": IntUniformDistribution(1, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        try:
            lgb = LGBMClassifier(device="gpu")
            lgb.fit(np.zeros((2, 2)), [0, 1])
            is_gpu_enabled = "gpu"
            del lgb
        except:
            try:
                lgb = LGBMClassifier(device="cuda")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                is_gpu_enabled = "cuda"
                del lgb
            except LightGBMError:
                is_gpu_enabled = False
                if globals_dict["gpu_param"] == "force":
                    raise RuntimeError(
                        f"LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                    )

        if is_gpu_enabled == "gpu":
            args["device"] = "gpu"
        elif is_gpu_enabled == "cuda":
            args["device"] = "cuda"

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=is_gpu_enabled,
        )


class CatBoostClassifierContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        try:
            import catboost
        except ImportError:
            logger.warning("Couldn't import catboost.CatBoostClassifier")
            self.active = False
            return

        if version.parse(catboost.__version__) < version.parse("0.23.2"):
            logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost.__version__}"
            )
            self.active = False
            return

        from catboost import CatBoostClassifier

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        args = {
            "random_state": globals_dict["seed"],
            "verbose": False,
            "thread_count": globals_dict["n_jobs_param"],
            "task_type": "GPU",
            "border_count": 32,
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

        tune_grid["depth"] = list(range(1, 9))
        tune_distributions["depth"] = (IntUniformDistribution(1, 8),)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="catboost",
            name="CatBoost Classifier",
            class_def=CatBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=True,
        )

BaggingClassifierContainer = pycaret.containers.models.local.classification.BaggingClassifierContainer

class CalibratedClassifierCVContainer(ClassifierContainer):
    def __init__(self, globals_dict: dict) -> None:
        logger = get_logger()
        np.random.seed(globals_dict["seed"])
        from sklearn.calibration import CalibratedClassifierCV
        from cuml.svm import SVC
        LinearSVC = SVC(kernel='linear')
        args = {
            "base_estimator": LinearSVC
        }
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="CalibratedCV",
            name="Calibrated Classifier CV",
            class_def=CalibratedClassifierCV,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=True,
        )


def get_all_model_containers(
        globals_dict: dict, raise_errors: bool = True
) -> Dict[str, ClassifierContainer]:
    return pycaret.containers.models.base_container.get_all_containers(
        globals(), globals_dict, ClassifierContainer, raise_errors
    )
