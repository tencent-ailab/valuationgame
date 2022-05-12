# Energy-Based Learning for Cooperative Games, with  Applications to Valuation Problems in Machine Learning

This repository contains code for the paper:  "Energy-Based Learning for Cooperative Games, with Applications to
Valuation Problems in Machine Learning".

## Run the experiments

### Synthetic experiments with FLID games

Run the file   ``synthetic/main_syn.ipynb`` directly.

### Data valuation

Run the following bash file:

```bash 
data_valuation/run_data_val.sh

```    

### Feature valuation

Run the ipython notebook ``feature_valuation/feature_val_example.ipynb``.

## Main depencencies (has been tested on Linux)

- python>=3.5
- shap==0.39.0
- tensorflow>2.0
- scikit-learn==0.24
- xgboost
- numba==0.53.1
- tqdm
- matplotlib

## Reference

:smile:If you find this repo is useful, please consider to cite our paper:

```
@inproceedings{
bian2022energybased,
title={Energy-Based Learning for Cooperative Games, with Applications to Valuation Problems in Machine Learning},
author={Yatao Bian and Yu Rong and Tingyang Xu and Jiaxiang Wu and Andreas Krause and Junzhou Huang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=xLfAgCroImw}
}
```  

## Disclaimer

This is not an officially supported Tencent product.



