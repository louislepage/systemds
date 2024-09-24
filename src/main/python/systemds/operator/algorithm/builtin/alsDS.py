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
# Autogenerated From : scripts/builtin/alsDS.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def alsDS(X: Matrix,
          **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     Alternating-Least-Squares (ALS) algorithm using a direct solve method for
     individual least squares problems (reg="L2"). This script computes an 
     approximate factorization of a low-rank matrix V into two matrices L and R.
     Matrices L and R are computed by minimizing a loss function (with regularization).
    
    
    
    :param X: Location to read the input matrix V to be factorized
    :param rank: Rank of the factorization
    :param reg: Regularization parameter, no regularization if 0.0
    :param maxi: Maximum number of iterations
    :param check: Check for convergence after every iteration, i.e., updating L and R once
    :param thr: Assuming check is set to TRUE, the algorithm stops and convergence is declared
        if the decrease in loss in any two consecutive iterations falls below this threshold;
        if check is FALSE thr is ignored
    :param seed: The seed to random parts of the algorithm
    :param verbose: If the algorithm should run verbosely
    :return: An m x r matrix where r is the factorization rank
    :return: An m x r matrix where r is the factorization rank
    """

    params_dict = {'X': X}
    params_dict.update(kwargs)
    
    vX_0 = Matrix(X.sds_context, '')
    vX_1 = Matrix(X.sds_context, '')
    output_nodes = [vX_0, vX_1, ]

    op = MultiReturn(X.sds_context, 'alsDS', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]

    return op
