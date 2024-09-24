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
# Autogenerated From : scripts/builtin/gmm.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def gmm(X: Matrix,
        **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     Gaussian Mixture Model training algorithm.
     There are four different types of covariance matrices
     i.e., VVV, EEE, VVI, VII and two initialization methods namely "kmeans" and "random".
    
    
    
    :param X: Dataset input to fit the GMM model
    :param n_components: Number of components to use in the Gaussian mixture model
    :param model: "VVV": unequal variance (full),each component has its own general covariance matrix
        "EEE": equal variance (tied), all components share the same general covariance matrix
        "VVI": spherical, unequal volume (diag), each component has its own diagonal
        covariance matrix
        "VII": spherical, equal volume (spherical), each component has its own single variance
    :param init_param: Initialization algorithm to use to initialize the gaussian weights, valid inputs are:
        "kmeans" or "random"
    :param iterations: Number of iterations
    :param reg_covar: Regularization parameter for covariance matrix
    :param tol: Tolerance value for convergence
    :param seed: The seed value to initialize the values for fitting the GMM.
    :return: The predictions made by the gaussian model on the X input dataset
    :return: Probability of the predictions given the X input dataset
    :return: Number of estimated parameters
    :return: Bayesian information criterion for best iteration
    :return: Fitted clusters mean
    :return: Fitted precision matrix for each mixture
    :return: The weight matrix:
        A matrix whose [i,k]th entry is the probability
        that observation i in the test data belongs to the kth class
    """

    params_dict = {'X': X}
    params_dict.update(kwargs)
    
    vX_0 = Matrix(X.sds_context, '')
    vX_1 = Matrix(X.sds_context, '')
    vX_2 = Scalar(X.sds_context, '')
    vX_3 = Scalar(X.sds_context, '')
    vX_4 = Matrix(X.sds_context, '')
    vX_5 = Matrix(X.sds_context, '')
    vX_6 = Matrix(X.sds_context, '')
    output_nodes = [vX_0, vX_1, vX_2, vX_3, vX_4, vX_5, vX_6, ]

    op = MultiReturn(X.sds_context, 'gmm', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]
    vX_2._unnamed_input_nodes = [op]
    vX_3._unnamed_input_nodes = [op]
    vX_4._unnamed_input_nodes = [op]
    vX_5._unnamed_input_nodes = [op]
    vX_6._unnamed_input_nodes = [op]

    return op
