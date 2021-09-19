import neat
import time
import numpy as np
import pandas as pd
#import pickle5 as pickle





l_df = pd.read_csv('/home/xprize/work/unmodified_pres.csv')
prescription_df = pd.concat([l_df,l_df])

modlist_a = ['C2_Workplace closing','C5_Close public transport','C7_Restrictions on internal movement']
modlist_b = [144,2345,17403]
modlist_u = [1,0,1]
modlist_val = [7,7,7]

for a in range (0,len(modlist_a)):
    if modlist_u[a]==1:
        prescription_df.loc[modlist_b[a],modlist_a[a]] = modlist_val[a]