# [HelloQlib](https://github.com/mczhuge/HelloQlib)

![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

## Introduction
I'm interested in program-based quantitative investing for a while. 
This is a project that I tried out with the **[Qlib](https://github.com/microsoft/qlib)** library, and it performs stock prediction experiments.
The goal is to make a model that is better than the standard transformer and its variants.
I made some attempts by introducing Probmask, TokenEmbedding inspired by **[Informer](https://github.com/zhouhaoyi/Informer2020)** that is popular to model long sequences. So I called the model "Simplified Informer". Besides, I use MAE loss to train the model. 

## Install

* Before installing ``HelloQlib`` from the source, readers should check the dependencies from **[Qlib](https://github.com/microsoft/qlib)**.

* Clone the repository and install ``Qlib`` as follows.
    ```bash
    git clone git@github.com:mczhuge/HelloQlib.git && cd HelloQlib/qlib
    pip install .
    ```

* Load and prepare data by running the following code:
  ```bash
  # get 1d data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # get 1min data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
  ```

As mentioned in Qlib, this dataset is created by public data collected by [crawler scripts](scripts/data_collector/), which have been released in
the same repository.


## Auto Quant Research Workflow
Qlib provides a tool named `qrun` to run the whole workflow automatically (including building dataset, training models, backtest and evaluation). You can start an auto quant research workflow and have a graphical reports analysis according to the following steps: 

1. Quant Research Workflow: Run  `qrun` with lightgbm workflow config ([workflow_config_lightgbm_Alpha158.yaml](examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) as following.
    ```bash
      cd examples  # Avoid running program under the directory contains `qlib`
      qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    If users want to use `qrun` under debug mode, please use the following command:
    ```bash
    python -m pdb qlib/workflow/cli.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    The result of `qrun` is as follows, please refer to [Intraday Trading](https://qlib.readthedocs.io/en/latest/component/backtest.html) for more details about the result. 

    ```bash

    'The following are analysis results of the excess return without cost.'
                           risk
    mean               0.000708
    std                0.005626
    annualized_return  0.178316
    information_ratio  1.996555
    max_drawdown      -0.081806
    'The following are analysis results of the excess return with cost.'
                           risk
    mean               0.000512
    std                0.005626
    annualized_return  0.128982
    information_ratio  1.444287
    max_drawdown      -0.091078
    ```
    Here are detailed documents for `qrun` and [workflow](https://qlib.readthedocs.io/en/latest/component/workflow.html).


## Licence
This is a quick play (and development) built upon Qlib. The right belongs to **[Qlib](https://github.com/microsoft/qlib)**：

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
