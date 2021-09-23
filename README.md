# PseUdeep:RNA pseuduridine sites identification with deep learning algorithm
##PseUdeep uses the following dependencies:
   * Python 3.6
   * numpy
   * scipy
   * scikit-learn
   * pandas

##Guiding principles:
**The dataset file contains five datasets, among which NH-990、NS-627、NM-944、H-200、S-200.Among them ,NH-990, NS-627, NM-944 are training datasets,H-200、S-200 are independent testing datasets
**Feature extraction：
  * KNFP_feature.py is the implementation of KNFP.
  * PNSP_train_feature is the implementation of PNSP on the train datasets.
  * PNSP_test_feature is the implementation of PNSP on the test datasets.
  * One_hot_feature is the implementation of one-hot encoding.
**Classifier:
  *PseUdeep.py is the implemention of PseUdeep
