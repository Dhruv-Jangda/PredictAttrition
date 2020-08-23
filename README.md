## PredictAttrition

A simple deep learning project pipelined as:
  * Data loading and preprocessing done in [Dask](https://dask.org/).
  * Model preparation and training done in [Mxnet](https://mxnet.apache.org/).
  * Training metric and model layer parameters visualization done in [Tensorboard](https://www.tensorflow.org/tensorboard) via [Mxboard](https://github.com/awslabs/mxboard).

The project has the following properties:
  1. **Scalable** - Due to :
      1. Hyperparameter and other configurations independent from project framework.
      2. All pipelines i.e. loading data, pre-processing data and model training independent of each other.
  2. **Ease of tuning** - Most of the hyperparameters are provided in the config.yaml file, thereby making model tuning easy.
  3. **Multiple modes** - Supports both *Imperative* and *Symbolic* mode of Mxnet modelling and training.
  

### Using Project

1. Setup python 3.6 environment with the following packages:
    1. Mxnet - The project supports CPU version as only.
    2. Dask
    3. Dask-Dataframe
    4. Mxboard
    5. Tensorboard
2. Clone the repository.
3. Double click *setup.py* to make supporting directories.
4. Double click *attrition.py* to run the project.

### Future releases
1. Support for Mxnet GPU version.
2. Support for Distributed Deep Learning. Though the data is handled in a distrbuted fashion as supported by Dask, but due to *symbolic* or *imperative* mode of training, the data has to be accumulated in memory for further processing.
