import neat
import time
import numpy as np
import pandas as pd
#import pickle5 as pickle

print(time.time())
x = np.random.randint(0,256,2)
str0 = bin(x[0]+256)[3:] + bin(x[1]+256)[3:]
ct = int(str0[0])+int(str0[1])+int(str0[2])+int(str0[3])+int(str0[4])+int(str0[5])+int(str0[6])+int(str0[7])+int(str0[8])+int(str0[9])+int(str0[10])+int(str0[11])+int(str0[12])+int(str0[13])+int(str0[14])+int(str0[15])



PRESCRIPTORS_FILE = 'neat-checkpoint-17'
checkpoint = neat.Checkpointer.restore_checkpoint(PRESCRIPTORS_FILE)

readFile = open('CovModel_modfile_1611897238.txt','r')
sepfile = readFile.read().split('\n')
Xseq = []
for a in range (1,len(sepfile),2):
    nx = sepfile[a].split(',')
    temp_seq=[]
    for b in range (0,150000):
        temp_seq.append(int(nx[b]))
    Xseq.append(temp_seq)
    
    
pres_df = pd.read_csv('/home/xprize/work/prescriptions/presA_2500.csv')

categories = ['C1_School closing','C2_Workplace closing','C3_Cancel public events','C4_Restrictions on gatherings','C5_Close public transport','C6_Stay at home requirements','C7_Restrictions on internal movement','C8_International travel controls','H1_Public information campaigns','H2_Testing policy','H3_Contact tracing','H6_Facial Coverings']

pert_df = pres_df.copy()

awesome = np.random.randint(0,7,len(pres_df))


for a in range (0,len(categories)):
    for b in range (0,len(pert_df)):
        if np.random.randint(0,2)==1:
            pert_df.loc[b,categories[a]] = 7
            
pres_df['awesomeness'] = awesome

print(pres_df)
print(pert_df)

pres_df = pert_df.copy()

print(pres_df)
print(pert_df)


AllLO=[]
Readfile=open('/home/xprize/work/words_219k_s3514.txt',encoding='latin-1')
Lines=Readfile.read().split('\n')
for line in range(9,len(Lines)-1):
    items=Lines[line].split('\t')
    AllLO.append(items[1])

LO_bank1 = AllLO[0:65536]
LO_bank2 = AllLO[65536:131072]
LO_bank3 = AllLO[131072:196608]
#print(len(LO_bank1),len(LO_bank2),len(LO_bank3))


pert_df = pert_df[['H2_Testing policy', 'C1_School closing']]

print(pert_df)
