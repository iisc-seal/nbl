# License

NeuralBugLocator (NBL in short) is available under the Apache License, version 2.0. Please see the LICENSE file for details.

# Reference

[Rahul Gupta](https://sites.google.com/site/csarahul/), Aditya Kanade, Shirish Shevade. ["Neural Attribution for Semantic Bug-Localization in Student Programs"](https://papers.nips.cc/paper/9358-neural-attribution-for-semantic-bug-localization-in-student-programs), NeurIPS 2019.

# Dataset

The [dataset](https://sites.google.com/site/csarahul/nbl-dataset.db.tar.gz) used to train and evaluate NBL is available under Apache 2.0, courtesy [Prof. Amey Karkare](https://www.cse.iitk.ac.in/users/karkare/) and his research group. It was collected from an introductory programming course at Indian Institute of Technology, Kanpur, India using a programming tutoring system called [Prutor](https://www.cse.iitk.ac.in/users/karkare/prutor/). If you use this dataset for your research, kindly give due credits to both Prutor and NBL.

# Running the tool

If you are using `Ubuntu 16.04.6 LTS` and `gcc version 5.4.0 20160609` (not tested on other distributions) and have conda installed, you can run `source init.sh` which creates a new virtual environment called `nbl` and installs all the dependencies in it.
Furthermore, it downloads and extracts the student submission data for you into the required directory structure.
Otherwise, run `source init.sh` and manually fix any failing steps.

To reproduce the results given in the paper, we have provided our processed training and validation data along with model checkpoints. Note that due to some changes made to anonymize the dataset, the results produced will be similar to those reported in the paper but may not exactly match.
To run your own experiments, use the following notebooks/scripts in the following order.

1. `generate_datasets.ipynb`

2. `train.py` for example, say: `python train.py data/network_inputs/bugloc-11-11/ data/checkpoints/bugloc-11-11/`

3. `nbl.ipynb`

4. `baselines/spectrum_based_baselines.ipynb`

5. `baselines/syntactic_diff_based_baseline.ipynb`
