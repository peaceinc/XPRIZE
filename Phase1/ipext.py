# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:25:59 2021

@author: Danny
"""
import sys

incsv = sys.argv[1]
outcsv = sys.argv[2]

mo = [6,7,8,9]
DIM = [30,31,31,30]
year = [2021,2021,2021,2021]


strLs=[]
for a in range (0,len(mo)):
    for b in range (1,DIM[a]+1):
        strLs.append('%d-%02d-%02d'%(year[a],mo[a],b))
        
newlines = []
        
readFile = open('%s'%incsv,'r')
sepfile = readFile.read().split('\n')

Header = sepfile[0]

x_dt=[]
for a in range (1,len(sepfile)-1):
    xandy = sepfile[a].split(',')
    Country = xandy[0]
    Region = xandy[1]
    dt = xandy[2]
    
    if a==len(sepfile)-2:
        NextCountry = 'Petoria'
        NextRegion = 'fatesville'
    else:
        NextCountry = sepfile[a+1].split(',')[0]
        NextRegion = sepfile[a+1].split(',')[1]
        
    
    
    if Country==NextCountry and Region==NextRegion:
        x_dt.append(dt)
        newlines.append(sepfile[a])
    else:
        x_dt.append(dt)
        startpos = len(Country)+len(Region)+2
        endpos = startpos+10
        repfile = sepfile[a]
        for b in range (0,len(strLs)):
            if strLs[b] not in x_dt:
                line = '%s%s%s'%(repfile[0:startpos],strLs[b],repfile[endpos:])
                newlines.append(line)
        
        x_dt=[]
                
outfile = open('%s'%outcsv,'w')
outfile.write('%s\n'%Header)
for a in range (0,len(newlines)):
    outfile.write('%s\n'%newlines[a])
outfile.close()