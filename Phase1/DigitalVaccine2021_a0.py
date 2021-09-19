
from __future__ import print_function, division
import csv
from urllib.request import urlopen as uReq
import numpy as np
import scipy.stats
#import geopandas
import matplotlib.pyplot as plt
import sys
import serial
from serial.tools import list_ports
import time


from pylab import polyfit

from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.impute import SimpleImputer
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from qboost import WeakClassifiers, QBoostClassifier, QboostPlus

from sklearn.model_selection import train_test_split

import pickle


#26 day distr
#BEFORE RUN A0... remove these notes and also change the DATE start/stop and other params. hope have number for all regions recent days.


UseTrueRNG = True
HALO = True
NEDspeed = 250#max = 50000
TurboSpeed = 250# max = 400000

NormThres = np.pi/8
SpThres = np.pi/2

outfile_path = 'C:/Users/Aslan/Phase1Sandbox'
starttime = time.time()


pCountry=[]
pRegion=[]
pDtstr=[]
pCovHalo=[]
pCov=[]
readFile = open('%s/predictions/formal_2.csv'%outfile_path,'r')
sepfile = readFile.read().split('\n')
for a in range (1,len(sepfile)-1):
    xandy = sepfile[a].split(',')
    pCountry.append(xandy[0])
    pRegion.append(xandy[1])
    pDtstr.append(xandy[2])
    pCovHalo.append(float(xandy[3]))
    pCov.append(float(xandy[4]))
    


readFile = open('%s/OxRegions2.txt'%outfile_path,'r')
sepfile = readFile.read().split('\n')

Countries=[]
Regions=[]
Special=[]
strox= []
for a in range (0,len(sepfile)):
    xandy = sepfile[a].split(',')
    Countries.append(xandy[0])
    Regions.append(xandy[1])
    Special.append(int(xandy[2]))
    strox.append('%s - %s'%(xandy[0],xandy[1]))




#37 -> 39 was error. Re-run for initializing A0.
#block in 7 days, %err cmp across regions.
#success measured by how many regions are hits vs misses.
#these hits vs misses in prv matter. what else A0 for prescription phase?
    
LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
uClient = uReq(LATEST_DATA_URL)
page_html = str(uClient.read())
uClient.close()    
    
    
sepfile = page_html.split('\\r\\n')
AbsError = []
Hits = []
UsableRegions = []
UsableIdx = []
for a in range (0,len(Countries)):
    
    x_obs=[]
    x_cov=[]
    x_halo=[]
    
    for b in range (2,len(sepfile)-1):
        xandy = sepfile[b].split(',')
        xandyM1 = sepfile[b-1].split(',')
        ctry = xandy[0]
        ste = xandy[2]
        B = xandy[5]
        dt = B[0:4]+'-'+B[4:6]+'-'+B[6:]
        if xandy[39]=='' or xandyM1[39]=='':
            numc = float("NaN")
        else:
            numc = int(xandy[39])-int(xandyM1[39])#so that its new per day instead of cumulative.
        
        
        if Regions[a]=='all':
            if ctry==Countries[a] and (dt in pDtstr) and ste=='':
                x_obs.append(numc)
        else:
            if ctry==Countries[a] and (dt in pDtstr) and ste==Regions[a]:
                x_obs.append(numc)
    
    
    #len check these 3
    for b in range (0,len(pCountry)):
        if pCountry[b]==Countries[a] and ((pRegion[b]==Regions[a]) or (pRegion[b]=='' and Regions[a]=='all')):
            x_cov.append(pCov[b])
            x_halo.append(pCovHalo[b])
            
    #print (a,len(x_obs),len(x_cov),len(x_halo))
    
    CovErr = []
    HaloErr = []
    
    for b in range (0,len(x_obs)-(len(x_obs)%7),7):
        if np.sum(x_obs[b:b+7])>0:
            CovErr.append(np.abs(np.sum(x_cov[b:b+7])-np.sum(x_obs[b:b+7]))/np.sum(x_obs[b:b+7]))
            HaloErr.append(np.abs(np.sum(x_halo[b:b+7])-np.sum(x_obs[b:b+7]))/np.sum(x_obs[b:b+7]))
        
    #print(CovErr,HaloErr)
        
    CxPower = np.nanmean(CovErr)-np.nanmean(HaloErr)
    
    if len(x_obs)>0 and len(x_cov)>0 and (np.sum(x_halo)>0 or np.sum(x_cov)>0):
    
        AbsError.append(CxPower)
        if CxPower>0:
            Hits.append(1)
        else:
            Hits.append(0)
            
        UsableRegions.append('%s / %s'%(Countries[a],Regions[a]))
        UsableIdx.append(a)
        
    print('evaluating previous data ... %s / %s'%(Countries[a],Regions[a]))
    
Score = np.sum(Hits)
print('%d hits of %d usable regions'%(Score,len(Hits)))

cutoff = np.nanmedian(AbsError)
TrueHits = []
for a in range (0,len(Hits)):
    if AbsError[a]>cutoff:
        TrueHits.append(1)
    else:
        TrueHits.append(0)
    

#for a in range (0,len(Regions)):
#    print(Countries[a],Regions[a],prv_results[a])

    

save_chi=[]

def Stream2chisq(stream):#turns single stream of bytes into bit-based "chi-square" values. stream should be multiple of 12 ... 75000 works well.
    chi=[]
    for a in range (0,len(stream),25):
        bitct = 0
        for b in range (0,25):
            strnode = str(bin(256+int(stream[a+b])))[3:]
            bitct += (int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7]))
        chi.append((bitct-100)**2)
        save_chi.append((bitct-100)**2)
    return chi
    

readFile = open('%s/text_files/CovModel_modfile_1622680884.txt'%outfile_path,'r')
sepfile = readFile.read().split('\n')
#prv_predicted=[]
prv_rng=[]
for a in range (0,len(sepfile)-1):
    if a in UsableIdx:
        xandy = sepfile[a].split(',')
        strchk = '%s - %s'%(xandy[75001],xandy[75002])
        if strchk in strox:
            xx=[]
            for b in range (0,75000):
                xx.append(int(xandy[b]))
            achi = Stream2chisq(xx)
            prv_rng.append(achi)
            #prv_predicted.append(float(xandy[-1]))


print(len(prv_rng),len(TrueHits),"should match")
chimeanprv = np.nanmean(save_chi)
chistdprv = np.nanstd(save_chi)
    

    
X_train, y_train = prv_rng, TrueHits
    
NUM_READS = 3000
NUM_WEAK_CLASSIFIERS = 35
# lmd = 0.5
TREE_DEPTH = 3

# define sampler
dwave_sampler = DWaveSampler(solver={'qpu': True})
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)

N_train = len(X_train)

print("\n======================================")
print("Train#: %d" %(N_train))
print('Num weak classifiers:', NUM_WEAK_CLASSIFIERS)
print('Tree depth:', TREE_DEPTH)


# input: dataset X and labels y (in {+1, -1}

# Preprocessing data
# imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()     # standardize features
normalizer = preprocessing.Normalizer()     # normalize samples

# X = imputer.fit_transform(X)    
X_train = scaler.fit_transform(X_train)
X_train = normalizer.fit_transform(X_train)
# X_test = imputer.fit_transform(X_test)



## Adaboost
print('\nAdaboost')
    
#print('fitting...')

clf = AdaBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS)
# scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)
file = open('%s/A01_%d.model'%(outfile_path,starttime),'wb')
pickle.dump(clf,file)
file.close()
    



#file = open('K:/my2Clf.model','rb')
#clf = pickle.load(file)
#file.close()

######


    




outfile = open('%s/CovModel_%d.txt'%(outfile_path,starttime),'w')
modfile = open('%s/CovModel_modfile_%d.txt'%(outfile_path,starttime),'w')


if UseTrueRNG==True:

    
    
    ports=dict()  
    ports_avaiable = list(list_ports.comports())
    
    
    rngcomports = []
    turbocom = None
    
    for temp in ports_avaiable:
        if HALO==True:
        	if temp[1].startswith("TrueRNG"):
        		if 'pro' in temp[1]:
        			print ('found pro')
        			turbocom = str(temp[0])
        		else:
        			print('Found:           ' + str(temp))
        			rngcomports.append(str(temp[0]))
        else:
        	if temp[1].startswith("TrueRNG"):
        		print ('found device')
        		turbocom = str(temp[0])
            
    if HALO==True:
        ser = []            
        for a in range(0,len(rngcomports)):
        	ser.append (serial.Serial(port=rngcomports[a],timeout=10))           
    turboser= (serial.Serial(port=turbocom,timeout=10)) 
    
    
               
    #print('Using com port:  ' + str(rng1_com_port))
    #print('Using com port:  ' + str(rng2_com_port))
    #print('==================================================')
    sys.stdout.flush()
    
    if HALO==True:
        for a in range(0,len(rngcomports)):
        	if(ser[a].isOpen() == False):
        		ser[a].open()
        
        	ser[a].setDTR(True)
        	ser[a].flushInput()
    if turboser.isOpen()==False:
        turboser.open()
    turboser.setDTR(True)
    turboser.flushInput()
    
    
    
    
    sys.stdout.flush()

def CohSampMain(params,Zthres,minruns):
    TotalRuns=0
    Zval = 0
    bitct=[]
    pct = []
    allnodes=[]
    for a in range (0,params):
        pct.append([])
    bads = np.zeros(params)
    while np.abs(Zval)<Zthres or TotalRuns<minruns:
        turboser.flushInput()
        supernode = turboser.read(TurboSpeed)
        
        for b in range (0,len(supernode)):
            outfile.write('%d,'%(supernode[b]))
            allnodes.append(supernode[b])
            strnode = str(bin(256+int(supernode[b])))[3:]
            bitct.append(int(strnode[0])+int(strnode[1])+int(strnode[2])+int(strnode[3])+int(strnode[4])+int(strnode[5])+int(strnode[6])+int(strnode[7]))
        outfile.write('%d,T\n'%(int(time.time()*1000)))
        
        
        for a in range(0,params):
            if HALO==True:
                try:
                    ser[a%len(ser)].flushInput()
                    node = ser[a%len(ser)].read(NEDspeed)
                except:
                    node = []
            else:
                node = turboser.read(NEDspeed)
            #print (a,len(node),TotalRuns)
            while len(node)==0:
                print('BAD READ ON %s ... removing'%rngcomports[a%len(ser)])
                ser.remove(ser[a%len(ser)])
                bads[a] += 1
                try:
                    ser[a%len(ser)].flushInput()
                    node = ser[a%len(ser)].read(NEDspeed)
                except:
                    node = []
            
            for mm in range (0,NEDspeed):
                outfile.write('%d,'%(node[mm]))
                strnum = bin(256+node[mm])[3:]
                pct[a].append((int(strnum[0]) + int(strnum[1]) + int(strnum[2]) + int(strnum[3]) + int(strnum[4]) + int(strnum[5]) + int(strnum[6]) + int(strnum[7])))
            outfile.write('%d,%s\n'%(int(time.time()*1000),rngcomports[a%len(ser)]))
        
        
        if TotalRuns < 300:
            NedVal = np.sum(bitct)
            TotalRuns += 1
            #print(bitct)
        else:
            TotalRuns = 300
            NedVal = np.sum(bitct[-(300*TurboSpeed):])
            #print(bitct[-60:])
        
        EX = NedVal-(TotalRuns*TurboSpeed*8*0.5)
        snpq = (TotalRuns*TurboSpeed*8*0.25)**0.5
        #print(TotalRuns,NedVal,EX,snpq)
        Zval = EX/snpq
        #print(Zval)
        
        Z=[]
        N = TotalRuns*NEDspeed*8
        for a in range (0,params):
            if TotalRuns < 300:
                NedVal_x = np.sum(pct[a])
            else:
                NedVal_x = np.sum(pct[a][-(300*NEDspeed):])
            Z.append((NedVal_x - (N*0.5)) / ((N*0.25)**0.5))
        
        time.sleep(0.2)
    #print(Z)
    #print(pct)
    #print(N)
    return Z,allnodes[(-minruns*TurboSpeed):]

#red = geopandas.datasets.get_path('c_10nv20')
#world = geopandas.read_file(red)
#rworld = world.iloc
    


ult_pert=[]
OnlyNodesLat=[]
OnlyNodesLon=[]
OnlyNodesPert=[]
NodeCt = 0
for a in range (0,len(Regions)):
    
    
    
    if Special[a]==0:
        T = NormThres
        print ('working on %s - %s, normal region'%(Countries[a],Regions[a]))
    else:
        T = SpThres
        print ('working on %s - %s, specialty region'%(Countries[a],Regions[a]))
    #sx = np.random.randint(0,2,1000)
    #Pur = (np.sum(sx)-500)/((1000*0.25)**0.5)
    
    y_test_pred = 0
    while y_test_pred < 1:
        sx = CohSampMain(8,T,300)
        print('node complete')
        
        #Pur = scipy.stats.chi2.sf(np.sum(np.array(sx[0])**2),8)
    
        Pur0 = sx[0][0]
        Pur1 = sx[0][1]
        Pur2 = sx[0][2]
        Pur3 = sx[0][3]
        Pur4 = sx[0][4]
        Pur5 = sx[0][5]
        Pur6 = sx[0][6]
        Pur7 = sx[0][7]
        
        sxx = sx[1]
        
        X_test = Stream2chisq(sxx)
        X_test = (np.array(X_test)-chimeanprv)/chistdprv
        X_test = [X_test]
        X_test = normalizer.fit_transform(X_test)
        
        hypotheses_ada = clf.estimators_
        y_test_pred = clf.predict(X_test)
        print(y_test_pred)
        if y_test_pred < 1:
            print('not enough consciousness, re-doing node')
    
    for c in range (0,len(sxx)):
        modfile.write('%d,'%(sxx[c]))
    modfile.write('%d,%s,%s,%f,%f,%f,%f,%f,%f,%f,%f\n'%(time.time()*1000,Countries[a],Regions[a],Pur0,Pur1,Pur2,Pur3,Pur4,Pur5,Pur6,Pur7))
        
    
        


outfile.close()
modfile.close()

