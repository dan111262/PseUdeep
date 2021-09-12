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

def PSP(pospath, negpath, k):
    posseq = read_fasta(pospath)
    negseq = read_fasta(negpath)
    positive = []
    for pos in posseq:
        positive.append(str(pos))
        
    negative = []
    for neg in negseq:
        negative.append(str(neg))
   
    p_num = len(positive)
    n_num = len(negative)
    lp = len(positive[0])
    ln = len(negative[0])
        
    nucleotides = ['A', 'C', 'G', 'U']
    
    if k == 1 :
        nuc = [n1 for n1 in nucleotides]
        order = {}
        for i in range(len(nuc)):
            order[nuc[i]] = i
        
        matrix_po = CalculateMatrix(positive, order, 1)
        matrix_ne = CalculateMatrix(negative, order, 1)

        F1 = matrix_po/p_num
        F2 = matrix_ne/n_num       
        F = F1 - F2
    
        poscode = []
        for sequence in positive:  
            for j in range(len(sequence)):                
                po_number = F[j][order[sequence[j:j+1]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((p_num,lp))
        
        negcode = []    
        for sequence in negative:    
            for i in range(len(sequence)):
                ne_number = F[i][order[sequence[i:i+1]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((n_num,ln))
        
    elif k == 2:
        dnuc = [n1 + n2  for n1 in nucleotides for n2 in nucleotides]
        order = {}
        for i in range(len(dnuc)):
            order[dnuc[i]] = i
        
        matrix_po = CalculateMatrix(positive, order, 2)
        matrix_ne = CalculateMatrix(negative, order, 2)
        
        F1 = matrix_po/p_num
        F2 = matrix_ne/n_num
        
        F = F1 - F2
        
        poscode = []
        for sequence in positive:  
            for j in range(len(sequence)-1):
                po_number = F[j][order[sequence[j: j+2]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((p_num,lp-1))
         
        negcode = []    
        for sequence in negative:    
            for j in range(len(sequence)-1):
                ne_number = F[j][order[sequence[j: j+2]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((n_num,ln-1))
        
    else:
        tnuc = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
        order = {}
        for i in range(len(tnuc)):
            order[tnuc[i]] = i
        
        matrix_po = CalculateMatrix(positive, order, 3)
        matrix_ne = CalculateMatrix(negative, order, 3)
        
        F1 = matrix_po/p_num
        F2 = matrix_ne/n_num
        F = F1 - F2

        poscode = []
        for sequence in positive:  
            for j in range(len(sequence) - 2):
                po_number = F[j][order[sequence[j: j+3]]]
                poscode.append(po_number)  
        poscode = np.array(poscode)
        poscode = poscode.reshape((p_num,lp-2))
         
        negcode = []    
        for sequence in negative:    
            for j in range(len(sequence) - 2):
                ne_number = F[j][order[sequence[j: j+3]]]
                negcode.append(ne_number)  
        negcode = np.array(negcode)
        negcode = negcode.reshape((n_num,ln-2))   
        
    return poscode, negcode

def read_fasta(datapath):
    sequences = list(SeqIO.parse(datapath, "fasta"))
    sequence = []
    for i in range(len(sequences)):
        sequence.append(sequences[i].seq)
    return sequence

def extract_features(pospath, negpath):
    PSNP_pos, PSNP_neg = np.array(PSP(pospath, negpath, 1))
    PSNP = np.concatenate((PSNP_pos,PSNP_neg), axis=0)
    # PSDP_pos, PSDP_neg = np.array(PSP(pospath, negpath, 2))
    # PSDP = np.concatenate((PSDP_pos,PSDP_neg), axis=0)
    # PSTP_pos, PSTP_neg = np.array(PSP(pospath, negpath, 3))
    # PSTP = np.concatenate((PSTP_pos,PSTP_neg), axis=0)
    # posseq = read_fasta(pospath)
    # negseq = read_fasta(negpath)
    feature_vector = PSNP
    return feature_vector

Mpt = 'E:/PseUdeep_master/data/PSNP_S/P_627.fasta'
Mnt = 'E:/PseUdeep_master/data/PSNP_S/N_627.fasta'
mouse_feature = extract_features(Mpt, Mnt)
pd.DataFrame(mouse_feature).to_csv("E:/PseUdeep_master/feature/PSNP/S_train_PSNP.csv",header=None,index=False)



