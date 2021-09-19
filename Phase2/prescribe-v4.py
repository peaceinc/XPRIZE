# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import datetime

import neat
import time

# Function imports from utils
from utils import add_geo_id
from utils import get_predictions
from utils import load_ips_file
from utils import prepare_historical_df

# Constant imports from utils
from utils import CASES_COL
from utils import IP_COLS
from utils import IP_MAX_VALUES
from utils import PRED_CASES_COL


# Path to file containing neat prescriptors. Here we simply use a
# recent checkpoint of the population from train_prescriptor.py,
# but this is likely not the most complementary set of prescriptors.
# Many approaches can be taken to generate/collect more diverse sets.
# Note: this set can contain up to 10 prescriptors for evaluation.
PRESCRIPTORS_FILE = '/home/xprize/work/neat-checkpoint-69'

# Number of days the prescriptors look at in the past.
NB_LOOKBACK_DAYS = 14

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 3

# Number of days to fix prescribed IPs before changing them.
# This could be a useful toggle for decision makers, who may not
# want to change policy every day. Increasing this value also
# can speed up the prescriptor, at the cost of potentially less
# interesting prescriptions.
ACTION_DURATION = 15


AllLO=[]
Readfile=open('/home/xprize/work/words_219k_s3514.txt',encoding='latin-1')
Lines=Readfile.read().split('\n')
for line in range(9,len(Lines)-1):
    items=Lines[line].split('\t')
    AllLO.append(items[1])




readFile = open('/home/xprize/work/CovModel_modfile_1612027429.txt','r')
sepfile = readFile.read().split('\n')
Xseq = []
for a in range (1,len(sepfile),2):
    nx = sepfile[a].split(',')
    temp_seq=[]
    for b in range (0,150000):
        temp_seq.append(int(nx[b]))
    Xseq.append(temp_seq)


def HALO(num,maxP,Hcall,Hcell,cat):
    action = 0
    
    if Hcall==-1:
        Hcall = int(Hcell/(ACTION_DURATION*2*236*12))+18
        Hcell = Hcell%(ACTION_DURATION*2*236*12)
    
    if(Hcall>=len(Xseq)):
        pert = np.random.randint(0,65536)
        print("HALO Overdrawn")
    else:
        pert = ((Xseq[Hcall][Hcell])*256) + ((Xseq[Hcall][Hcell+1]))
    if pert<8:
        action = -1
    if pert>=65528:
        action = 1
    if (action == -1 and num>=1) or (action == 1 and num<maxP):
        num += action
    if (action == -1 and num<1) or (action==1 and num>=maxP):
        print("modified active but at limit")
        
    LO_idx = ((cat%3)*65536)+pert
    wrd = AllLO[LO_idx]
    
    timestr = str(time.time())
    str0 = bin((Xseq[Hcall][Hcell])+256)[3:] + bin((Xseq[Hcall][Hcell+1])+256)[3:]
    ones = int(str0[0])+int(str0[1])+int(str0[2])+int(str0[3])+int(str0[4])+int(str0[5])+int(str0[6])+int(str0[7])+int(str0[8])+int(str0[9])+int(str0[10])+int(str0[11])+int(str0[12])+int(str0[13])+int(str0[14])+int(str0[15])

    Z = np.abs(ones-8)

    if ones==8:
        symbol = '.0'
    if ones<8:
        symbol = '.- %d'%Z
    if ones>8:
        symbol = '.+ %d'%Z

    SR_time = timestr + symbol
    
    return num,wrd,SR_time


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    ult_words = []
    ult_halotime=[]
    modlist_a=[]
    modlist_b=[]
    modlist_val=[]
    modlist_u=[]
    
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Load the past IPs data
    print("Loading past IPs data...")
    past_ips_df = load_ips_file(path_to_prior_ips_file)
    geos = past_ips_df['GeoID'].unique()
    print(len(geos),"LEN GEOS")

    # Load historical data with basic preprocessing
    print("Loading historical data...")
    df = prepare_historical_df()

    # Restrict it to dates before the start_date
    df = df[df['Date'] <= start_date]

    # Create past case data arrays for all geos
    past_cases = {}
    for geo in geos:
        geo_df = df[df['GeoID'] == geo]
        past_cases[geo] = np.maximum(0, np.array(geo_df[CASES_COL]))

    # Create past ip data arrays for all geos
    past_ips = {}
    for geo in geos:
        geo_df = past_ips_df[past_ips_df['GeoID'] == geo]
        past_ips[geo] = np.array(geo_df[IP_COLS])

    # Fill in any missing case data before start_date
    # using predictor given past_ips_df.
    # Note that the following assumes that the df returned by prepare_historical_df()
    # has the same final date for all regions. This has been true so far, but relies
    # on it being true for the Oxford data csv loaded by prepare_historical_df().
    last_historical_data_date_str = df['Date'].max()
    last_historical_data_date = pd.to_datetime(last_historical_data_date_str,
                                               format='%Y-%m-%d')
    if last_historical_data_date + pd.Timedelta(days=1) < start_date:
        print("Filling in missing data...")
        missing_data_start_date = last_historical_data_date + pd.Timedelta(days=1)
        missing_data_start_date_str = datetime.strftime(missing_data_start_date,
                                                           format='%Y-%m-%d')
        missing_data_end_date = start_date - pd.Timedelta(days=1)
        missing_data_end_date_str = datetime.strftime(missing_data_end_date,
                                                           format='%Y-%m-%d')
        pred_df = get_predictions(missing_data_start_date_str,
                                  missing_data_end_date_str,
                                  past_ips_df)
        pred_df = add_geo_id(pred_df)
        for geo in geos:
            geo_df = pred_df[pred_df['GeoID'] == geo].sort_values(by='Date')
            pred_cases_arr = np.array(geo_df[PRED_CASES_COL])
            past_cases[geo] = np.append(past_cases[geo], pred_cases_arr)
    else:
        print("No missing data.")

    # Gather values for scaling network output
    ip_max_values_arr = np.array([IP_MAX_VALUES[ip] for ip in IP_COLS])

    # Load prescriptors
    checkpoint = neat.Checkpointer.restore_checkpoint(PRESCRIPTORS_FILE)
    prescriptors = list(checkpoint.population.values())[:NB_PRESCRIPTIONS]
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-prescriptor')

    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in geos:
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

    # Generate prescriptions
    prescription_dfs = []
    #Eout = open('/home/xprize/work/prescriptions/dataframes.txt','w')
    dfCt = 2900
    BaseModIdx = 0
    print(dfCt)
    HaloCallNum = 0
    for prescription_idx, prescriptor in enumerate(prescriptors):
        print("Generating prescription", prescription_idx, "...")
        print(time.time())

        # Create net from genome
        net = neat.nn.FeedForwardNetwork.create(prescriptor, config)

        # Set up dictionary for keeping track of prescription
        df_dict = {'CountryName': [], 'RegionName': [], 'Date': []}
        for ip_col in sorted(IP_MAX_VALUES.keys()):
            df_dict[ip_col] = []

        # Set initial data
        eval_past_cases = deepcopy(past_cases)
        eval_past_ips = deepcopy(past_ips)

        # Generate prescriptions iteratively, feeding resulting
        # predictions from the predictor back into the prescriptor.
        action_start_date = start_date
        while action_start_date <= end_date:

            # Get prescription for all regions
            for geo in geos:

                # Prepare input data. Here we use log to place cases
                # on a reasonable scale; many other approaches are possible.
                X_cases = np.log(eval_past_cases[geo][-NB_LOOKBACK_DAYS:] + 1)
                X_ips = eval_past_ips[geo][-NB_LOOKBACK_DAYS:]
                X_costs = geo_costs[geo]
                X = np.concatenate([X_cases.flatten(),
                                    X_ips.flatten(),
                                    X_costs])

                # Get prescription
                prescribed_ips = net.activate(X)

                # Map prescription to integer outputs
                prescribed_ips = (prescribed_ips * ip_max_values_arr).round()

                # Add it to prescription dictionary for the full ACTION_DURATION
                country_name, region_name = geo.split('__')
                if region_name == 'nan':
                    region_name = np.nan
                for date in pd.date_range(action_start_date, periods=ACTION_DURATION):
                    if date > end_date:
                        break
                    date_str = date.strftime("%Y-%m-%d")
                    df_dict['CountryName'].append(country_name)
                    df_dict['RegionName'].append(region_name)
                    df_dict['Date'].append(date_str)
                    for ip_col, prescribed_ip in zip(IP_COLS, prescribed_ips):
                        df_dict[ip_col].append(prescribed_ip)

            # Create dataframe from prescriptions
            pres_df = pd.DataFrame(df_dict)

            # Make prediction given prescription for all countries
            pre_prediction_time = int(time.time())
            pred_df = get_predictions(start_date_str, date_str, pres_df)
            post_prediction_time = int(time.time())
            print('predictions took %d seconds'%(post_prediction_time-pre_prediction_time))
            
            #Eout.write('%s\n'%action_start_date)
            #Eout.write('%s\n\n\n'%pres_df)
            #Eout.write('%s\n\n\n'%pred_df)
            


            
            
            pred_df = pred_df[pred_df.CountryName != 'Kiribati']
            pred_df = pred_df[pred_df.CountryName != 'Malta']
            pred_df = pred_df[pred_df.CountryName != 'Turkmenistan']
            pred_df = pred_df[pred_df.CountryName != 'Tonga']
            pred_df = pred_df[pred_df.RegionName != 'Alberta']
            pred_df = pred_df[pred_df.RegionName != 'British Columbia']
            pred_df = pred_df[pred_df.RegionName != 'Manitoba']
            pred_df = pred_df[pred_df.RegionName != 'New Brunswick']
            pred_df = pred_df[pred_df.RegionName != 'Newfoundland and Labrador']
            pred_df = pred_df[pred_df.RegionName != 'Nova Scotia']
            pred_df = pred_df[pred_df.RegionName != 'Northwest Territories']
            pred_df = pred_df[pred_df.RegionName != 'Nunavut']
            pred_df = pred_df[pred_df.RegionName != 'Ontario']
            pred_df = pred_df[pred_df.RegionName != 'Prince Edward Island']
            pred_df = pred_df[pred_df.RegionName != 'Quebec']
            pred_df = pred_df[pred_df.RegionName != 'Saskatchewan']
            pred_df = pred_df[pred_df.RegionName != 'Yukon']
            pred_df = pred_df[pred_df.RegionName != 'Virgin Islands']
            
            
            pred_df.reset_index(drop=True, inplace=True)
            pred_df['Unnamed: 0'] = pred_df.index
            
            ixx = -1*len(geos)*ACTION_DURATION

            


            categories = ['C1_School closing','C2_Workplace closing','C3_Cancel public events','C4_Restrictions on gatherings','C5_Close public transport','C6_Stay at home requirements','C7_Restrictions on internal movement','C8_International travel controls','H1_Public information campaigns','H2_Testing policy','H3_Contact tracing','H6_Facial Coverings']
            categories_max = [3,3,2,4,2,3,2,4,2,3,2,4]

            cov = pred_df['PredictedDailyNewCases'].iloc[ixx:].sum()
            n_cost = 0
            for a in range (0,len(categories)):
                #n_cost += pres_df[categories[a]].iloc[ixx:].sum()
                for b in range (0,ixx*-1):
                    ixx_x = b + ixx
                    ixx_m = b%ACTION_DURATION
                    n_cost += ((pres_df[categories[a]].iloc[ixx_x])*(cost_df.loc[ixx_m, categories[a]]))
                
            NormalPower = cov*n_cost*-1
            print(NormalPower,cov,n_cost,"NO HALO")
            
            #duplicate prescription dataframe.
            pert_df = pres_df.copy()
            
            HaloCellNum = 0
            countPert = 0
            for a in range (0,len(categories)):
                for b in range (len(pert_df)+ixx,len(pert_df)):
                    cell = pres_df[categories[a]].iloc[b]
                    pert,word,halo_t = HALO(cell,categories_max[a],HaloCallNum,HaloCellNum,a)
                    if pert!=cell:
                        print('modified %s %d'%(categories[a],b))
                        modlist_a.append(categories[a])
                        modlist_b.append(b+BaseModIdx)
                        modlist_val.append(pert)
                        countPert += 1
                    HaloCellNum += 2
                    pert_df.loc[b,categories[a]] = pert
                    ult_words.append(word)
                    ult_halotime.append(halo_t)
            HaloCallNum += 1
            pre_prediction_time = int(time.time())
            
            pertpred_df = get_predictions(start_date_str, date_str, pert_df)
            
            #pertpred_df = pertpred_df[pertpred_df.CountryName != 'Kiribati']
            #pertpred_df = pertpred_df[pertpred_df.CountryName != 'Malta']
            #pertpred_df = pertpred_df[pertpred_df.CountryName != 'Turkmenistan']
            #pertpred_df = pertpred_df[pertpred_df.CountryName != 'Tonga']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Alberta']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'British Columbia']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Manitoba']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'New Brunswick']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Newfoundland and Labrador']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Nova Scotia']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Northwest Territories']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Nunavut']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Ontario']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Prince Edward Island']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Quebec']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Saskatchewan']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Yukon']
            #pertpred_df = pertpred_df[pertpred_df.RegionName != 'Virgin Islands']
            
            
            post_prediction_time = int(time.time())
            print('predictions took %d seconds'%(post_prediction_time-pre_prediction_time))
            
            cov = pertpred_df['PredictedDailyNewCases'].iloc[ixx:].sum()
            n_cost = 0
            for a in range (0,len(categories)):
                #n_cost += pert_df[categories[a]].iloc[ixx:].sum()
                for b in range (0,ixx*-1):
                    ixx_x = b + ixx
                    ixx_m = b%ACTION_DURATION
                    n_cost += ((pert_df[categories[a]].iloc[ixx_x])*(cost_df.loc[ixx_m, categories[a]]))
            HaloPower = cov*n_cost*-1
            print(HaloPower,cov,n_cost,"WITH HALO")
            
            pred_df.to_csv('/home/xprize/work/prescriptions/pred_%d.csv'%dfCt)
            pres_df.to_csv('/home/xprize/work/prescriptions/presA_%d.csv'%dfCt)
            pert_df.to_csv('/home/xprize/work/prescriptions/pert_%d.csv'%dfCt)
            pertpred_df.to_csv('/home/xprize/work/prescriptions/pertpred_%d.csv'%dfCt)
            
            if HaloPower > NormalPower:
                pred_df = pertpred_df.copy()
                pres_df = pert_df.copy()
                print("HALO TURBOCHARGE ASSESSED")
                countPertV = 1
            else:
                print("HALO TURBOCHARGE DID NOT FIND BETTER SOLUTION :/")
                countPertV = 0
                
            for a in range (0,countPert):
                modlist_u.append(countPertV)
            
            print(len(cost_df),len(pert_df),"SHOULD BE ==236")#HALO on uk going to zero...needs eval.

            
            
            pres_df.to_csv('/home/xprize/work/prescriptions/presB_%d.csv'%dfCt)
            
            # Update past data with new days of prescriptions and predictions
            pres_df = add_geo_id(pres_df)
            pred_df = add_geo_id(pred_df)
            
            for date in pd.date_range(action_start_date, periods=ACTION_DURATION):
                if date > end_date:
                    break
                date_str = date.strftime("%Y-%m-%d")
                new_pres_df = pres_df[pres_df['Date'] == date_str]
                new_pred_df = pred_df[pred_df['Date'] == date_str]
                for geo in geos:
                    geo_pres = new_pres_df[new_pres_df['GeoID'] == geo]
                    geo_pred = new_pred_df[new_pred_df['GeoID'] == geo]
                    # Append array of prescriptions
                    pres_arr = np.array([geo_pres[ip_col].values[0] for
                                         ip_col in IP_COLS]).reshape(1,-1)
                    eval_past_ips[geo] = np.concatenate([eval_past_ips[geo], pres_arr])

                    # It is possible that the predictor does not return values for some regions.
                    # To make sure we generate full prescriptions, this script continues anyway.
                    # This should not happen, but is included here for robustness.
                    if len(geo_pred) != 0:
                        eval_past_cases[geo] = np.append(eval_past_cases[geo],
                                                         geo_pred[PRED_CASES_COL].values[0])

            pres_df.to_csv('/home/xprize/work/prescriptions/presC_%d.csv'%dfCt)
            
            # Move on to next action date
            dfCt += 1
            action_start_date += pd.DateOffset(days=ACTION_DURATION)

        # Add prescription df to list of all prescriptions for this submission
        BaseModIdx += len(pres_df)
        pres_df['PrescriptionIndex'] = prescription_idx
        prescription_dfs.append(pres_df)
    #Eout.close()
    # Combine dfs for all prescriptions into a single df for the submission
    prescription_df = pd.concat(prescription_dfs)
    prescription_df = prescription_df.drop(columns='GeoID')

    # Create the output directory if necessary.
    output_dir = os.path.dirname(output_file_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
        
    prescription_df.to_csv('/home/xprize/work/unmodified_pres.csv', index=False)
        
    OGlen = len(prescription_df)

    ff_df = pd.concat([prescription_df,prescription_df,prescription_df,prescription_df])
    ff_df.reset_index(drop=True, inplace=True)

    daysran = int(len(ff_df)/(len(geos)*12))
    print(daysran)

    CutRows = len(geos)*2*daysran
    ff_df = ff_df[:-CutRows]

    for a in range (0,len(ff_df)):
        ff_df.loc[a,'PrescriptionIndex'] = int(a/(daysran*len(geos)))

        
    #modify prescription output by brute force
    for a in range (0,len(modlist_a)):
        if modlist_u[a]==1:
            ff_df.loc[modlist_b[a],modlist_a[a]] = modlist_val[a]
    
    HaloCellNum = 0
    for a in range (0,len(categories)):
        for b in range (OGlen,len(ff_df)):
            cell = ff_df[categories[a]].iloc[b]
            pert,word,halo_t = HALO(cell,categories_max[a],-1,HaloCellNum,a)
            if pert!=cell:
                print('modified %d %d'%(a,b))
                ff_df.loc[b,categories[a]] = pert
            HaloCellNum += 2
            
            ult_words.append(word)
            ult_halotime.append(halo_t)

            
    print(len(ff_df),len(ult_words),"should be 12 times")
    ff_df['C1_words'] = ult_words[0:len(ff_df)]
    ff_df['C2_words'] = ult_words[len(ff_df):2*len(ff_df)]
    ff_df['C3_words'] = ult_words[2*len(ff_df):3*len(ff_df)]
    ff_df['C4_words'] = ult_words[3*len(ff_df):4*len(ff_df)]
    ff_df['C5_words'] = ult_words[4*len(ff_df):5*len(ff_df)]
    ff_df['C6_words'] = ult_words[5*len(ff_df):6*len(ff_df)]
    ff_df['C7_words'] = ult_words[6*len(ff_df):7*len(ff_df)]
    ff_df['C8_words'] = ult_words[7*len(ff_df):8*len(ff_df)]
    ff_df['H1_words'] = ult_words[8*len(ff_df):9*len(ff_df)]
    ff_df['H2_words'] = ult_words[9*len(ff_df):10*len(ff_df)]
    ff_df['H3_words'] = ult_words[10*len(ff_df):11*len(ff_df)]
    ff_df['H6_words'] = ult_words[11*len(ff_df):12*len(ff_df)]
    
    ff_df['C1_time'] = ult_halotime[0:len(ff_df)]
    ff_df['C2_time'] = ult_halotime[len(ff_df):2*len(ff_df)]
    ff_df['C3_time'] = ult_halotime[2*len(ff_df):3*len(ff_df)]
    ff_df['C4_time'] = ult_halotime[3*len(ff_df):4*len(ff_df)]
    ff_df['C5_time'] = ult_halotime[4*len(ff_df):5*len(ff_df)]
    ff_df['C6_time'] = ult_halotime[5*len(ff_df):6*len(ff_df)]
    ff_df['C7_time'] = ult_halotime[6*len(ff_df):7*len(ff_df)]
    ff_df['C8_time'] = ult_halotime[7*len(ff_df):8*len(ff_df)]
    ff_df['H1_time'] = ult_halotime[8*len(ff_df):9*len(ff_df)]
    ff_df['H2_time'] = ult_halotime[9*len(ff_df):10*len(ff_df)]
    ff_df['H3_time'] = ult_halotime[10*len(ff_df):11*len(ff_df)]
    ff_df['H6_time'] = ult_halotime[11*len(ff_df):12*len(ff_df)]
    
    w_p = 1/196608
    w_q = 1-w_p
    w_n = len(ff_df)*12
    
    #CtWords = []
    #ZWords=[]
    #for a in range (0,len(ult_words)):
    #    xct = ult_words.count(ult_words[a])
    #    CtWords.append(xct)
    #    ZWords.append((xct - (w_n*w_p))/((w_n*w_p*w_q)**0.5))
    #sCMW=[xf for _,xf in sorted (zip(CtWords,ult_words),reverse=True)]
    

    
    #ff_df['TopWords'] = sCMW[0:len(ff_df)]
    #ff_df['TopWords_Zscore'] = ZWords[0:len(ff_df)]
    
    # Save to a csv file
    #prescription_df.to_csv(output_file_path, index=False)
    ff_df.to_csv(output_file_path, index=False)
    print('Prescriptions saved to', output_file_path)
    print(time.time())

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prior_ips_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")
