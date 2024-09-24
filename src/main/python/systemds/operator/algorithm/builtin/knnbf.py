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
# Autogenerated From : scripts/builtin/knnbf.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def knnbf(X: Matrix,
          T: Matrix,
          **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This script implements KNN (K Nearest Neighbor) algorithm.
    
    
    
    :param X: ---
    :param T: ---
    :param k_value: ---
    :return: ---
    """

    params_dict = {'X': X, 'T': T}
    params_dict.update(kwargs)
    return Matrix(X.sds_context,
        'knnbf',
        named_input_nodes=params_dict)
