#!/bin/bash
#-------------------------------------------------------------
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
#-------------------------------------------------------------
#
#Runs systemds and python implementation multiple times and stores phis to files

data_file="${1:-../data/runtimes_permutation_test.csv}"
instances=5

adult_data_sysds_str="data_dir=../data/adult/ X_bg_path=Adult_X.csv B_path=Adult_W.csv metadata_path=Adult_meta.csv model_type=multiLogRegPredict"
adult_data_python_str="--data-dir=../data/adult/ --data-x=Adult_X.csv --model-type=multiLogReg"

census_data_sysds_str="data_dir=../data/census/ X_bg_path=census_xTrain.csv B_path=census_bias.csv metadata_path=census_dummycoding_meta.csv model_type=l2svmPredict"
census_data_python_str="--data-dir=../data/census/ --data-x=census_xTrain.csv --data-y=census_yTrain_corrected.csv --model-type=l2svm"

exp_type_array=("adult_linlogreg" "census_l2svm")

for samples in {1..200..20}; do
  for permutations in {1..20..2}; do
	      for exp_type in "${exp_type_array[@]}"; do

	        if [ "$exp_type" = "adult_linlogreg" ]; then
	            data_str=$adult_data_sysds_str
              py_str=$adult_data_python_str
          elif [ "$exp_type" = "census_l2svm" ]; then
              data_str=$census_data_sysds_str
              py_str=$census_data_python_str
          else
              echo "Exp type unknown: $exp_type"
              exit 1
          fi

          echo "Running $exp_type for $permutations permutrations and $samples samples..."
          #python
          python ./shap-permutation.py ${py_str} --n-permutations=${permutations} --n-instances=${instances} --n-samples=${samples} --silent

          #by-row
          systemds ./shapley-permutation-experiment.dml -nvargs ${data_str} n_permutations=${permutations} integration_samples=${samples} rows_to_explain=${instances} write_to_file=0 execution_policy=by-row

        done
    done
done
