# Module: containers.models.anomaly
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of anomaly models. The `anomaly` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging

import pycaret.containers.base_container
import pycaret.internal.cuml_wrappers
from pycaret.containers.models.deprecated.anomaly import (
    ABODAnomalyContainer,
    AnomalyContainer,
    CBLOFAnomalyContainer,
    COFAnomalyContainer,
    HBOSAnomalyContainer,
    IForestAnomalyContainer,
    KNNAnomalyContainer,
    LOFAnomalyContainer,
    MCDAnomalyContainer,
    OCSVMAnomalyContainer,
    PCAAnomalyContainer,
    SODAnomalyContainer,
    SOSAnomalyContainer,
)
from pycaret.internal.distributions import *


_DEFAULT_N_ANOMALYS = 4


class CumlABODAnomalyContainer(ABODAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlCBLOFAnomalyContainer(CBLOFAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlCOFAnomalyContainer(COFAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlIForestAnomalyContainer(IForestAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlHBOSAnomalyContainer(HBOSAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlKNNAnomalyContainer(KNNAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlLOFAnomalyContainer(LOFAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlOCSVMAnomalyContainer(OCSVMAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlPCAAnomalyContainer(PCAAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlMCDAnomalyContainer(MCDAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlSODAnomalyContainer(SODAnomalyContainer):
    def __init__(self):
        super().__init__()


class CumlSOSAnomalyContainer(SOSAnomalyContainer):
    def __init__(self):
        super().__init__()


def get_all_model_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, AnomalyContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, AnomalyContainer, raise_errors
    )
