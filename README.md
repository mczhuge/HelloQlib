#HelloQlib

![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

## Introduction
This is a project that I tried with the **[Qlib](https://github.com/microsoft/qlib)** library, and I use it for stock prediction. The goal is to make a model that is better than the standard transformer and its variants.
I made some attempts to introduce Probmask, TokenEmbedding inspired by **[Informer](https://github.com/zhouhaoyi/Informer2020)**. So I called the model "Simplified Informer". Besides, I use MAE loss to train the model. 

# Install
### Dependancy

* Before installing ``HelloQlib`` from source, readers shoud check the dependancies from **[Qlib](https://github.com/microsoft/qlib)**.
    ```bash
    Python = 3.7
    Numpy
    Cython
    ```
The above three are needed.

* Clone the repository and install ``Qlib`` as follows.
    ```bash
    git clone https://github.com/microsoft/qlib.git && cd qlib
    pip install .
    ```
  **Note**:  You can install Qlib with `python setup.py install` as well. But it is not the recommanded approach. It will skip `pip` and cause obscure problems. For example, **only** the command ``pip install .`` **can** overwrite the stable version installed by ``pip install pyqlib``, while the command ``python setup.py install`` **can't**.

**Tips**: If you fail to install `Qlib` or run the examples in your environment,  comparing your steps and the [CI workflow](.github/workflows/test.yml) may help you find the problem.

## Data Preparation
Load and prepare data by running the following code:
  ```bash
  # get 1d data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  # get 1min data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

  ```

This dataset is created by public data collected by [crawler scripts](scripts/data_collector/), which have been released in
the same repository.
Users could create the same dataset with it. [Description of dataset](https://github.com/microsoft/qlib/tree/main/scripts/data_collector#description-of-dataset)

*Please pay **ATTENTION** that the data is collected from [Yahoo Finance](https://finance.yahoo.com/lookup), and the data might not be perfect.
We recommend users to prepare their own data if they have a high-quality dataset. For more information, users can refer to the [related document](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*.

### Automatic update of daily frequency data (from yahoo finance)
  > This step is *Optional* if users only want to try their models and strategies on history data.
  > 
  > It is recommended that users update the data manually once (--trading_date 2021-05-25) and then set it to update automatically.
  >
  > For more information, please refer to: [yahoo collector](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)

  * Automatic update of data to the "qlib" directory each trading day(Linux)
      * use *crontab*: `crontab -e`
      * set up timed tasks:

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **script path**: *scripts/data_collector/yahoo/collector.py*

  * Manual update of data
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
      ```
      * *trading_date*: start of trading day
      * *end_date*: end of trading day(not included)


<!-- 
- Run the initialization code and get stock data:

  ```python
  import qlib
  from qlib.data import D
  from qlib.constant import REG_CN

  # Initialization
  mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
  qlib.init(mount_path=mount_path, region=REG_CN)

  # Get stock data by Qlib
  # Load trading calendar with the given time range and frequency
  print(D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2])

  # Parse a given market name into a stockpool config
  instruments = D.instruments('csi500')
  print(D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6])

  # Load features of certain instruments in given time range
  instruments = ['SH600000']
  fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
  print(D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head())
  ```
 -->

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
