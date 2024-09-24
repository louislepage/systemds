# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

# Autogenerated By   : src/main/python/generator/generator.py
# Autogenerated From : scripts/builtin/randomForestPredict.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def randomForestPredict(X: Matrix,
                        ctypes: Matrix,
                        M: Matrix,
                        **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This script implements random forest prediction for recoded and binned
     categorical and numerical input features.
    
    
    
    :param X: Feature matrix in recoded/binned representation
    :param y: Label matrix in recoded/binned representation,
        optional for accuracy evaluation
    :param ctypes: Row-Vector of column types [1 scale/ordinal, 2 categorical]
    :param M: Matrix M holding the learned trees (one tree per row),
        see randomForest() for the detailed tree representation.
    :param verbose: Flag indicating verbose debug output
    :return: Label vector of predictions
    """

    params_dict = {'X': X, 'ctypes': ctypes, 'M': M}
    params_dict.update(kwargs)
    return Matrix(X.sds_context,
        'randomForestPredict',
        named_input_nodes=params_dict)
