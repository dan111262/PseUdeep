from keras.layers import  Activation
from keras.layers import Dense, Convolution1D, Dropout, Input, Flatten,  AveragePooling1D,GRU
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import optimizers
import math
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import keras
import pandas as pd
import numpy as np
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
from model_Attention import  Capsule

def Twoclassfy_evalu1(y_test, y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
    for i in range(len(y_test)):
        if y_predict1[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict1[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 0:
            TN += 1
    # TP
    # TN
    # FP
    # FN
    TP += 1
    FP += 1
    FN += 1
    TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
    Precision = TP / (TP + FP)
    F1_score = (2 * Precision * Sn) / (Precision + Sn)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict1)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
    i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    print('TP', TP)
    print('FP', FP)
    print('FN', FN)
    print('TN', TN)

    print('Sn', Sn)
    print('Sp', Sp)
    print('ACC', Acc)
    print('Mcc', Mcc)
    print('Precision', Precision)
    print('F1_score', F1_score)
    print('AUC', aucs)

    result = [TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs]
    np.savetxt('E:/PseUdeep_master/Test_result.txt', result, delimiter=" ", fmt='%s')
    return Sn, Acc, Precision, F1_score, aucs

def read_train_feature(filepath):
    feature = pd.read_csv(filepath, header = None,index_col = False)
    feature = np.array(feature)
    return feature

def createModel():
    one_input = Input(shape=(31, 4), name='one_input')
    one = Convolution1D(filters=11, kernel_size=7, padding='same')(one_input)
    one = Convolution1D(filters=11, kernel_size=7, padding='same')(one)
    one = BatchNormalization(axis=-1)(one)
    one = Activation('relu')(one)

    profile_input = Input(shape=(31, 84), name='profile_input')
    profile = Convolution1D(filters=11, kernel_size=1, padding='same')(profile_input)
    profile = Convolution1D(filters=11, kernel_size=1, padding='same')(profile)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('relu')(profile)

    PSNP_input = Input(shape=(31, 1), name='PSNP_input')
    PSNP = Convolution1D(filters=11, kernel_size=1, padding='same')(PSNP_input)
    PSNP = Convolution1D(filters=11, kernel_size=1, padding='same')(PSNP)
    PSNP = BatchNormalization(axis=-1)(PSNP)
    PSNP = Activation('relu')(PSNP)

    mergeInput = Concatenate(axis=-1)([one,profile,PSNP])

    overallResult = AveragePooling1D(pool_size=5)(mergeInput)
    overallResult = Dropout(0.5)(overallResult)
    overallResult = Capsule(num_capsule=14, dim_capsule=41,
                            routings=3, share_weights=True)(overallResult)
    overallResult = Bidirectional(GRU(32, return_sequences=True))(overallResult)
    overallResult = Dropout(0.5)(overallResult)
    overallResult = SeqSelfAttention(
        attention_activation='sigmoid',
        name='Attention',
    )(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(128, activation='relu')(overallResult)
    overallResult = Dense(16, activation='relu')(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)

    return Model(inputs=[one_input,profile_input,PSNP_input], outputs=[ss_output])
#PSNP feature
filepath = "E:/PseUdeep_master/feature/PSNP/S_train_PSNP.csv"
x=read_train_feature(filepath)
PSNP_input = x.reshape((x.shape[0],31,1))
#one-hot feature
print('Loading data...')
X = np.load("E:/PseUdeep_master/feature/one-hot/S_627_onehot.npy")
one_input = X.reshape((X.shape[0],31,4))
#KNFP feature
train_profile = np.load('E:/PseUdeep_master/feature/KNFP/S_627_indataX.npy')
Y = np.array([1] *314 + [0] * 313,dtype='float32')
# Y= Y.reshape((Y.shape[0],1))
# Y = Y[:,0]

seed = 7
np.random.seed(seed)
KF=KFold(10, True)
for train_index, eval_index in KF.split(Y):
    train_X1 = train_profile[train_index]
    train_X2 = one_input[train_index]
    train_X3 = PSNP_input[train_index]
    train_y =Y[train_index]
    train_y = keras.utils.to_categorical(train_y, 2)
    eval_X1 = train_profile[eval_index]
    eval_X2 = one_input[eval_index]
    eval_X3 = PSNP_input[eval_index]
    eval_y = Y[eval_index]
    eval_y = keras.utils.to_categorical(eval_y, 2)

    model  = createModel()
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-08),loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])
    model.fit([train_X2,train_X1,train_X3],[train_y], epochs=200, batch_size=88,verbose=2,validation_data=([eval_X2,eval_X1,eval_X3],[eval_y]))
    score = model.evaluate([eval_X2,eval_X1,eval_X3],[eval_y])
    y_predict = model.predict([eval_X2,eval_X1,eval_X3])
    c1 = list(y_predict[:,0])
    c2 =eval_y[:,0]
    model.save_weights('zhuyili_S_weight.h5')
    # model.save('bay_PSNP_after_S.h5')


if __name__ == '__main__':
    Twoclassfy_evalu1(c2, c1)
