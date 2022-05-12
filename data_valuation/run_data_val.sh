#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:`pwd`"
# breast cancer dataset 1
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/breast_cancer \
                                              --train_cluster_number 16 \
                                              --cluster rand \
                                              --dataset breast_cancer \
                                              --tempe 0.1 \
                                              --output breast_cancer

# breast cancer dataset 2
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/breast_cancer2 \
                                              --train_cluster_number 14 \
                                              --cluster kmeans \
                                              --dataset breast_cancer \
                                              --tempe 0.5 \
                                              --output breast_cancer2


# digits dataset
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/digits \
                                              --train_cluster_number 12 \
                                              --cluster rand \
                                              --dataset digits \
                                              --tempe 0.5 \
                                              --output digits

# digits dataset 2
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/digits2 \
                                              --train_cluster_number 10 \
                                              --cluster kmeans \
                                              --dataset digits \
                                              --tempe 0.1 \
                                              --output digits2
# synthentic data without cluster
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/syn_clus20_sz1 \
                                              --train_cluster_number 20 \
                                              --tempe 0.5
# synthentic data with cluster
python data_valuation/main_data_val.py --tmp_dir data_valuation/example_results/syn_clus20_sz50_kmeans \
                                              --train_cluster_number 20 \
                                              --train_cluster_size 50 \
                                              --tempe 1

