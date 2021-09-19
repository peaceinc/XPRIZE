import neat
import time
import numpy as np
import pandas as pd
#import pickle5 as pickle

x = int((time.time()-1612339200)/(86400*3))
if x>7:
    x=7
print(x)

timestr = str(time.time())
x = np.random.randint(0,256,2)
str0 = bin(x[0]+256)[3:] + bin(x[1]+256)[3:]
ones = int(str0[0])+int(str0[1])+int(str0[2])+int(str0[3])+int(str0[4])+int(str0[5])+int(str0[6])+int(str0[7])+int(str0[8])+int(str0[9])+int(str0[10])+int(str0[11])+int(str0[12])+int(str0[13])+int(str0[14])+int(str0[15])

Z = np.abs(ones-8)

if ones==8:
    symbol = '.0'
if ones<8:
    symbol = '.- %d'%Z
if ones>8:
    symbol = '.+ %d'%Z

SR_time = timestr + symbol
print(SR_time)