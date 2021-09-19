#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.
# In[1]:


import os
import pandas as pd
import requests


# ### Expected output
# For each row that was provided in input, i.e. for each country, region and day, the output should contain an additional `PredictedDailyNewCases` column with the predicted number of cases for that day, region and country. It is possible to leave `PredictedDailyNewCases` empty or NaN, or to remove the row, in case no predition is available.

# In[ ]:


#EXAMPLE_OUTPUT_FILE = "../../../../2020-08-01_2020-08-04_predictions_example.csv"


# In[ ]:


#prediction_output_df = pd.read_csv(EXAMPLE_OUTPUT_FILE,
#                                   parse_dates=['Date'],
#                                   encoding="ISO-8859-1")


# In[ ]:


#prediction_output_df.head()


# ### Evaluation
# Predictions will be evaluated on a period of 4 weeks **after** submision against the actual daily change in confirmed cases reported by the [Oxford COVID-19 Government Response Tracker (OxCGRT)](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker).
# 
# The latest data, including the latest confirmed cases ('ConfirmedCases') can be find here: https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv

# In[ ]:


#DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
#df = pd.read_csv(DATA_URL,
#                 parse_dates=['Date'],
#                 encoding="ISO-8859-1",
#                 dtype={"RegionName": str,
#                        "RegionCode": str},
#                 error_bad_lines=False)


# In[ ]:


#df.sample(3)


# ### Daily change in confirmed cases
# The daily change in confirmed cases can be computed like this:

# In[ ]:


#df["DailyChangeConfirmedCases"] = df.groupby(["CountryName", "RegionName"]).ConfirmedCases.diff().fillna(0)


# For instance, for country **United States**, region **California**, the latest available changes in confirmed cases are:

# In[ ]:


#california_df = df[(df.CountryName == "United States") & (df.RegionName == "California")]


# In[ ]:


#california_df[["CountryName", "RegionName", "Date", "ConfirmedCases", "DailyChangeConfirmedCases"]].tail(5)


# # Training a model

# ## Copy the data locally

# In[2]:


# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
DATA_FILE = 'data/OxCGRT_latest.csv'

# Download the data set
data = requests.get(DATA_URL)

# Persist the data set locally in order to use it after submission to make predictions,
# as the sandbox won't have access to the internet anymore.
if not os.path.exists('data'):
    os.mkdir('data')
open(DATA_FILE, 'wb').write(data.content)


# ## Train

# In[3]:


# Reload the module to get the latest changes
import xprize_predictor
from importlib import reload
reload(xprize_predictor)
from xprize_predictor import XPrizePredictor
predictor = XPrizePredictor(None, DATA_FILE)


# In[4]:


get_ipython().run_cell_magic('time', '', 'predictor_model = predictor.train()')


# In[6]:


if not os.path.exists('models'):
    os.mkdir('models')
predictor_model.save_weights("models/trained_model_weights.h5")


# # Predicting using a trained model

# ## Load candidate model

# In[ ]:


#model_weights_file = "models/trained_model_weights.h5"


# In[ ]:


#predictor = XPrizePredictor(model_weights_file, DATA_FILE)


# ## Make prediction

# In[ ]:


#NPIS_INPUT_FILE = "../../../validation/data/2020-09-30_historical_ip.csv"
#start_date = "2020-08-01"
#end_date = "2020-08-31"


# In[ ]:


#get_ipython().run_cell_magic('time', '', 'preds_df = predictor.predict(start_date, end_date, NPIS_INPUT_FILE)')


# In[ ]:


#preds_df.head()


# # Validation
# This is how the predictor is going to be called during the competition.  
# !!! PLEASE DO NOT CHANGE THE API !!!

# In[ ]:


#get_ipython().system('python predict.py -s 2020-08-01 -e 2020-08-04 -ip ../../../validation/data/2020-09-30_historical_ip.csv -o predictions/2020-08-01_2020-08-04.csv')


# In[ ]:


#get_ipython().system('head predictions/2020-08-01_2020-08-04.csv')


# # Test cases
# We can generate a prediction file. Let's validate a few cases...

# In[ ]:


# Check the pediction file is valid
#import os
#from covid_xprize.validation.predictor_validation import validate_submission

#def validate(start_date, end_date, ip_file, output_file):
    # First, delete any potential old file
#    try:
#        os.remove(output_file)
#    except OSError:
#        pass
    
    # Then generate the prediction, calling the official API
#    get_ipython().system('python predict.py -s {start_date} -e {end_date} -ip {ip_file} -o {output_file}')
    
    # And validate it
#    errors = validate_submission(start_date, end_date, ip_file, output_file)
#    if errors:
#        for error in errors:
#            print(error)
#    else:
#        print("All good!")


# ## 4 days, no gap
# - All countries and regions
# - Official number of cases is known up to start_date
# - Intervention Plans are the official ones

# In[ ]:


#validate(start_date="2020-08-01",
#         end_date="2020-08-04",
#         ip_file="../../../validation/data/2020-09-30_historical_ip.csv",
#         output_file="predictions/val_4_days.csv")


# ## 1 month in the future
# - 2 countries only
# - there's a gap between date of last known number of cases and start_date
# - For future dates, Intervention Plans contains scenarios for which predictions are requested to answer the question: what will happen if we apply these plans?

# In[ ]:


#get_ipython().run_cell_magic('time', '', 'validate(start_date="2021-01-01",\n         end_date="2021-01-31",\n         ip_file="../../../validation/data/future_ip.csv",\n         output_file="predictions/val_1_month_future.csv")')


# ## 180 days, from a future date, all countries and regions
# - Prediction start date is 1 week from now. (i.e. assuming submission date is 1 week from now)  
# - Prediction end date is 6 months after start date.  
# - Prediction is requested for all available countries and regions.  
# - Intervention plan scenario: freeze last known intervention plans for each country and region.  
# 
# As the number of cases is not known yet between today and start date, but the model relies on them, the model has to predict them in order to use them.  
# This test is the most demanding test. It should take less than 1 hour to generate the prediction file.

# ### Generate the scenario

# In[ ]:


#from datetime import datetime, timedelta

#start_date = datetime.now() + timedelta(days=7)
#start_date_str = start_date.strftime('%Y-%m-%d')
#end_date = start_date + timedelta(days=180)
#end_date_str = end_date.strftime('%Y-%m-%d')
#print(f"Start date: {start_date_str}")
#print(f"End date: {end_date_str}")


# In[ ]:


#from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario, NPI_COLUMNS
#DATA_FILE = 'data/OxCGRT_latest.csv'
#latest_df = get_raw_data(DATA_FILE, latest=True)
#scenario_df = generate_scenario(start_date_str, end_date_str, latest_df, countries=None, scenario="Freeze")
#scenario_file = "predictions/180_days_future_scenario.csv"
#scenario_df.to_csv(scenario_file, index=False)
#print(f"Saved scenario to {scenario_file}")


# ### Check it

# In[ ]:


#get_ipython().run_cell_magic('time', '', 'validate(start_date=start_date_str,\n         end_date=end_date_str,\n         ip_file=scenario_file,\n         output_file="predictions/val_6_month_future.csv")')


# In[ ]:




