import pandas as pd
import numpy as np

def read_fasta_file():
    fh = open('E:/PseUdeep_master/data/S_627.txt', 'r')
    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))
    fh.close()
    matrix_data = np.array([list(e) for e in seq])
    print(matrix_data)
    print(len(matrix_data))
    return matrix_data

def extract_line(data_line):
    A=[0,0,0,1]
    U=[0,0,1,0]
    C=[0,1,0,0]
    G=[1,0,0,0]
    
    feature_representation={"A":A,"C":C,"G":G,"U":U }
    one_line_feature=[]
    for index,data in enumerate(data_line):
        if data in feature_representation.keys():
            one_line_feature.extend(feature_representation[data])
    return one_line_feature   

def feature_extraction(matrix_data):    
    final_feature_matrix=[extract_line(e) for e in matrix_data]
    return final_feature_matrix

matrix_data = read_fasta_file()
#print(matrix_data)
final_feature_matrix = feature_extraction(matrix_data)
#print(final_feature_matrix)
print(np.array(final_feature_matrix).shape)
pd.DataFrame(final_feature_matrix).to_csv('E:/PseUdeep_master/feature/one-hot/S_627_one_hot.csv',header=None,index=False)


final_feature_matrix1 = np.array(final_feature_matrix)
np.save("E:/PseUdeep_master/feature/one-hot/S_627_onehot.npy",final_feature_matrix1)








