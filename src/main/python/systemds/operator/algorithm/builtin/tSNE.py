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
# Autogenerated From : scripts/builtin/tSNE.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def tSNE(X: Matrix,
         **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This function performs dimensionality reduction using tSNE algorithm based on
     the paper: Visualizing Data using t-SNE, Maaten et. al.
    
     There exists a variant of t-SNE, implemented in sklearn, that first reduces the
     dimenisonality of the data using PCA to reduce noise and then applies t-SNE for
     further dimensionality reduction. A script of this can be found in the tutorials
     folder: scripts/tutorials/tsne/pca-tsne.dml
    
     For direct reference and tips on choosing the dimension for the PCA pre-processing,
     you can visit:
     https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py
     https://lvdmaaten.github.io/tsne/
    
    
    
    :param X: Data Matrix of shape
        (number of data points, input dimensionality)
    :param reduced_dims: Output dimensionality
    :param perplexity: Perplexity Parameter
    :param lr: Learning rate
    :param momentum: Momentum Parameter
    :param max_iter: Number of iterations
    :param tol: Tolerance for early stopping in gradient descent
    :param seed: The seed used for initial values.
        If set to -1 random seeds are selected.
    :param is_verbose: Print debug information
    :param print_iter: Intervals of printing out the L1 norm values. Parameter not relevant if
        is_verbose = FALSE.
    :return: Data Matrix of shape (number of data points, reduced_dims)
    """

    params_dict = {'X': X}
    params_dict.update(kwargs)
    return Matrix(X.sds_context,
        'tSNE',
        named_input_nodes=params_dict)
