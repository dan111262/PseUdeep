# PseUdeep:RNA pseuduridine sites identification with deep learning algorithm
##PseUdeep uses the following dependencies:
   * Python 3.6
   * numpy
   * scipy
   * scikit-learn
   * pandas
##Guiding principles:
**The dataset file contains five datasets, among which NH-990、NS-627、NM-944、H-200、S-200
**Feature extraction：
  * KNFP_feature.py is the implementation of KNFP.
  * PNSP_train_feature is the implementation of KNFP on the train datasets.
  * PNSP_test_feature is the implementation of KNFP on the test datasets.
  * One_hot_feature is the implementation of one-hot.
**Classifier:
  *PseUdeep.py is the implemention of PseUdeep
