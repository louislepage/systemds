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
# Autogenerated From : scripts/builtin/naiveBayes.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def naiveBayes(D: Matrix,
               C: Matrix,
               **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This builtin function implements a NaiveBayes classification.
    
    
    
    :param D: Input feature matrix of shape N x M
    :param C: Class label vector (positive integers) of shape N x 1.
    :param laplace: Laplace smoothing correction (prevent zero probabilities)
    :param verbose: Flag for verbose debug output
    :return: Class prior probabilities
    :return: Class conditional feature distributions
    """

    params_dict = {'D': D, 'C': C}
    params_dict.update(kwargs)
    
    vX_0 = Matrix(D.sds_context, '')
    vX_1 = Matrix(D.sds_context, '')
    output_nodes = [vX_0, vX_1, ]

    op = MultiReturn(D.sds_context, 'naiveBayes', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]

    return op
