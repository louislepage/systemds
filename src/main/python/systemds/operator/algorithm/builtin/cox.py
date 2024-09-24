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
# Autogenerated From : scripts/builtin/cox.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.utils.consts import VALID_INPUT_TYPES


def cox(X: Matrix,
        TE: Matrix,
        F: Matrix,
        R: Matrix,
        **kwargs: Dict[str, VALID_INPUT_TYPES]):
    """
     This script fits a cox Proportional hazard regression model.
     The Breslow method is used for handling ties and the regression parameters 
     are computed using trust region newton method with conjugate gradient 
     
    
    
    :param X: Location to read the input matrix X containing the survival data
        containing the following information
        1: timestamps
        2: whether an event occurred (1) or data is censored (0)
        3: feature vectors
    :param TE: Column indices of X as a column vector which contain timestamp
        (first row) and event information (second row)
    :param F: Column indices of X as a column vector which are to be used for
        fitting the Cox model
    :param R: If factors (categorical variables) are available in the input matrix
        X, location to read matrix R containing the start and end indices of
        the factors in X
        R[,1]: start indices
        R[,2]: end indices
        Alternatively, user can specify the indices of the baseline level of
        each factor which needs to be removed from X; in this case the start
        and end indices corresponding to the baseline level need to be the same;
        if R is not provided by default all variables are considered to be continuous
    :param alpha: Parameter to compute a 100*(1-alpha)% confidence interval for the betas
    :param tol: Tolerance ("epsilon")
    :param moi: Max. number of outer (Newton) iterations
    :param mii: Max. number of inner (conjugate gradient) iterations, 0 = no max
    :return: A D x 7 matrix M, where D denotes the number of covariates, with the following schema:
        M[,1]: betas
        M[,2]: exp(betas)
        M[,3]: standard error of betas
        M[,4]: Z
        M[,5]: P-value
        M[,6]: lower 100*(1-alpha)% confidence interval of betas
        M[,7]: upper 100*(1-alpha)% confidence interval of betas
    :return: Two matrices containing a summary of some statistics of the fitted model:
        1 - File S with the following format
        - row 1: no. of observations
        - row 2: no. of events
        - row 3: log-likelihood
        - row 4: AIC
        - row 5: Rsquare (Cox & Snell)
        - row 6: max possible Rsquare
        2 - File T with the following format
        - row 1: Likelihood ratio test statistic, degree of freedom, P-value
        - row 2: Wald test statistic, degree of freedom, P-value
        - row 3: Score (log-rank) test statistic, degree of freedom, P-value
    :return: Additionally, the following matrices are stored (needed for prediction)
        1- A column matrix RT that contains the order-preserving recoded timestamps from X
        2- Matrix XO which is matrix X with sorted timestamps
        3- Variance-covariance matrix of the betas COV
        4- A column matrix MF that contains the column indices of X with the baseline factors removed (if available)
    """

    params_dict = {'X': X, 'TE': TE, 'F': F, 'R': R}
    params_dict.update(kwargs)
    
    vX_0 = Matrix(X.sds_context, '')
    vX_1 = Matrix(X.sds_context, '')
    vX_2 = Matrix(X.sds_context, '')
    vX_3 = Matrix(X.sds_context, '')
    vX_4 = Matrix(X.sds_context, '')
    vX_5 = Matrix(X.sds_context, '')
    output_nodes = [vX_0, vX_1, vX_2, vX_3, vX_4, vX_5, ]

    op = MultiReturn(X.sds_context, 'cox', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]
    vX_2._unnamed_input_nodes = [op]
    vX_3._unnamed_input_nodes = [op]
    vX_4._unnamed_input_nodes = [op]
    vX_5._unnamed_input_nodes = [op]

    return op
