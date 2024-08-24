# Comprehensive Codebase

This repository contains a comprehensive codebase for the execution of Machine Learning (ML) tasks considering various input criteria.

**Prerequisites Installation**

Create a new virtual environment named `multiethnic`, activate it, and install the required dependencies as listed below:

Python 3.7 - https://www.python.org/downloads/release/python-370/

Numpy 1.21.5 - https://pypi.org/project/numpy/1.21.5/

Pandas 1.3.5 - https://pypi.org/project/pandas/1.3.5/

Scipy 1.7.3 - https://pypi.org/project/scipy/1.7.3/

Scikit-learn 1.0.2 - https://pypi.org/project/scikit-learn/1.0.2/

Theano 1.0.3 - https://pypi.org/project/Theano/1.0.3/

Tensorflow 1.13.1 - https://pypi.org/project/tensorflow/1.13.1/

Tensorflow-estimator 1.13.0 - https://pypi.org/project/tensorflow-estimator/1.13.0/

Tensorboard 1.13.1 - https://pypi.org/project/tensorboard/1.13.1/

Keras 2.2.4 - https://pypi.org/project/keras/2.2.4/

Keras-applications 1.0.8 - https://pypi.org/project/Keras-Applications/

Keras-preprocessing 1.1.0 - https://pypi.org/project/Keras-Preprocessing/1.1.0/

Pytorch 1.10.2 - https://pypi.org/project/torch/1.10.2/

Lasagne 0.2.dev1 - https://github.com/Lasagne/Lasagne

Xlrd 1.1.0 - https://pypi.org/project/xlrd/1.1.0/

openpyxl - https://pypi.org/project/openpyxl/

**Running the codes**

STEP 1 - Download the required datasets using the download links provided in the respective subfolders under Dataset/EssentialData/ folder.

STEP 2 - Activate the virtual environment named `multiethnic`, Run `main.py` with input arguments using the following command:

```
python main.py --data_Category <data_category> --omicsConfiguration <omics_configuration> --DDP_group <ddp_group> --cancer_type <cancer_type> --omics_feature <omics_feature> --endpoint <endpoint> --years <years> --features_count <features_count> --FeatureMethod <FeatureMethod> --AutoencoderSettings <AutoencoderSettings>
```

Please make sure to read the details on which input argument is required for which condition in `main.py` file.

## Global paths

Before running the scripts, ensure to verify the global paths of the dataset and the current folder in which you are working. In the Python code `main.py`, the default paths are defined as follows:

    folderISAAC = 'GenderBidirectionalTransfer/'
    if os.path.exists(folderISAAC)!=True:
        folderISAAC = './'

## Acknowledgement

This work has been supported by NIH R01 grant.


## Contact

For any queries, please contact:

Prof. Yan Cui (ycui2@uthsc.edu)

Dr. Teena Sharma (tee.shar6@gmail.com)

