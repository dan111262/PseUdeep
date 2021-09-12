import itertools
import numpy as np 
from Bio import SeqIO
import pickle as pkl
import pandas as pd 

def CalculateMatrix(data, order, k):
    if k == 1:
        matrix = np.zeros((len(data[0]), 4))
        for i in range(len(data[0])): # position
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+1]]] += 1
    elif k == 2:
        matrix = np.zeros((len(data[0]) - 1, 16))
        for i in range(len(data[0]) - 1): # position
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+2]]] += 1
    else:
        matrix = np.zeros((len(data[0]) - 2, 64))
        for i in range(len(data[0]) - 2): # position
            for j in range(len(data)):
                matrix[i][order[data[j][i:i+3]]] += 1           
    return matrix

def PSP(train_pospath, train_negpath, test_pospath, test_negpath, k):
    train_posseq = read_fasta(train_pospath)
    train_negseq = read_fasta(train_negpath)
    train_positive = []
    for pos in train_posseq:
        train_positive.append(str(pos))
        
    train_negative = []
    for neg in train_negseq:
        train_negative.append(str(neg))
   
    train_p_num = len(train_positive)
    train_n_num = len(train_negative)
    
    test_posseq = read_fasta(test_pospath)
    test_negseq = read_fasta(test_negpath)
    test_positive = []
    for pos in test_posseq:
        test_positive.append(str(pos))
        
    test_negative = []
    for neg in test_negseq:
        test_negative.append(str(neg))
   
    test_p_num = len(test_positive)
    test_n_num = len(test_negative)
    
    test_lp = len(test_positive[0])
    test_ln = len(test_negative[0])
        
    nucleotides = ['A', 'C', 'G', 'U']
    
    if k == 1 :
        nuc = [n1 for n1 in nucleotides]
        order = {}
        for i in range(len(nuc)):
            order[nuc[i]] = i
        
        matrix_po = CalculateMatrix(train_positive, order, 1)
        matrix_ne = CalculateMatrix(train_negative, order, 1)

        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        F = F1 - F2

        poscode = []
        for sequence in test_positive:  
            for j in range(len(sequence)):                
                po_number = F[j][order[sequence[j:j+1]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((test_p_num,test_lp))
        
        negcode = []    
        for sequence in test_negative:    
            for i in range(len(sequence)):
                ne_number = F[i][order[sequence[i:i+1]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((test_n_num,test_ln))
        
    elif k == 2:
        dnuc = [n1 + n2  for n1 in nucleotides for n2 in nucleotides]
        order = {}
        for i in range(len(dnuc)):
            order[dnuc[i]] = i
        
        matrix_po = CalculateMatrix(train_positive, order, 2)
        matrix_ne = CalculateMatrix(train_negative, order, 2)
        
        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        
        F = F1 - F2
        
        poscode = []
        for sequence in test_positive:  
            for j in range(len(sequence)-1):                
                po_number = F[j][order[sequence[j:j+2]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((test_p_num,test_lp-1))
        
        negcode = []    
        for sequence in test_negative:    
            for i in range(len(sequence)-1):
                ne_number = F[i][order[sequence[i:i+2]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((test_n_num,test_ln-1))
        
    else:
        tnuc = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
        order = {}
        for i in range(len(tnuc)):
            order[tnuc[i]] = i
        
        matrix_po = CalculateMatrix(train_positive, order, 3)
        matrix_ne = CalculateMatrix(train_negative, order, 3)
        
        F1 = matrix_po/train_p_num
        F2 = matrix_ne/train_n_num       
        
        F = F1 - F2
        
        poscode = []
        for sequence in test_positive:  
            for j in range(len(sequence)-2):                
                po_number = F[j][order[sequence[j:j+3]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((test_p_num,test_lp-2))
        
        negcode = []    
        for sequence in test_negative:    
            for i in range(len(sequence)-2):
                ne_number = F[i][order[sequence[i:i+3]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((test_n_num,test_ln-2))
        
    return poscode, negcode

def read_fasta(datapath):
    sequences = list(SeqIO.parse(datapath, "fasta"))
    sequence = []
    for i in range(len(sequences)):
        sequence.append(sequences[i].seq)
    return sequence

def extract_test_features(train_pospath, train_negpath, test_pospath, test_negpath):
    PSNP_pos, PSNP_neg = np.array(PSP(train_pospath, train_negpath, test_pospath, test_negpath, 1))
    PSNP = np.concatenate((PSNP_pos,PSNP_neg), axis=0)
    # PSDP_pos, PSDP_neg = np.array(PSP(train_pospath, train_negpath, test_pospath, test_negpath, 2))
    # PSDP = np.concatenate((PSDP_pos,PSDP_neg), axis=0)
    # PSTP_pos, PSTP_neg = np.array(PSP(train_pospath, train_negpath, test_pospath, test_negpath, 3))
    # PSTP = np.concatenate((PSTP_pos,PSTP_neg), axis=0)
    # posseq = read_fasta(test_pospath)
    # negseq = read_fasta(test_negpath)
    # feature_vector = np.concatenate((PSNP, PSDP, PSTP, Kmer1, PCPseDNC1, PseEIIP1), axis=1)
    feature_vector = PSNP
    return feature_vector

train_pospath_mouse = 'E:/PseUdeep_master/data/PSNP_S/P_627.fasta'
train_negpath_mouse = 'E:/PseUdeep_master/data/PSNP_S/N_627.fasta'

test_pospath_mouse = 'E:/PseUdeep_master/data/PSNP_S/P_100.fasta'
test_negpath_mouse = 'E:/PseUdeep_master/data/PSNP_S/N_100.fasta'

mouse_feature_ind = extract_test_features(train_pospath_mouse, train_negpath_mouse, test_pospath_mouse, test_negpath_mouse)
pd.DataFrame(mouse_feature_ind).to_csv("E:/PseUdeep_master/feature/PSNP/S_test_PSNP.csv",header=None,index=False)







