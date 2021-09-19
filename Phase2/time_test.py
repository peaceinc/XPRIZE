import neat
import time
import numpy as np
import pandas as pd
#import pickle5 as pickle


prescription_df = pd.read_csv('/home/xprize/work/unmodified_pres.csv')




AllLO=[]
Readfile=open('/home/xprize/work/words_219k_s3514.txt',encoding='latin-1')
Lines=Readfile.read().split('\n')
for line in range(9,len(Lines)-1):
    items=Lines[line].split('\t')
    AllLO.append(items[1])

LO_bank = AllLO[0:196608]

w_n = 90*235*12*1

ult_words = []
for a in range (0,w_n):
    idx = np.random.randint(0,196608)
    ult_words.append(LO_bank[idx])

    
w_p = 1/196608
w_q = 1-w_p

    


ult_words_eval = ult_words[0:int(len(ult_words)/3)]
if len(ult_words_eval)>=196608:
    l_eval = 196608
else:
    l_eval = len(ult_words_eval)
        
print(l_eval)
    
CtWords = []
ZWords=[]
Bird = []

start = time.time()

for a in range (0,l_eval):
    xct = ult_words_eval.count(AllLO[a])
    CtWords.append(xct)
    ZWords.append((xct - (w_n*w_p))/((w_n*w_p*w_q)**0.5))
    Bird.append(AllLO[a])
    if a%1000==0:
        print(a)
sCMW=[xf for _,xf in sorted (zip(CtWords,Bird),reverse=True)]
sZ=sorted(ZWords,reverse=True)


fin = time.time()
print('took %d seconds'%(int(fin-start)))

outfile = open('/home/xprize/work/Words_out11.txt','w')
for a in range (0,len(sCMW)):
    outfile.write('%s,%d\n'%(sCMW[a],sZ[a]))
outfile.close()

outfile = open('/home/xprize/work/Words_out12.txt','w')
for a in range (0,len(Bird)):
    outfile.write('%s,%d\n'%(Bird[a],CtWords[a]))
outfile.close()