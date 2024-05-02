#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:08:50 2024

@author: kk423
"""

import pandas as pd
import numpy as np
import random

def datagenerator(file,total_rows):

# Calculate the number of unique values in column 'A'
    num_unique_values = len(file['ID'].unique())

# Calculate the number of times each unique value should be repeated
    target_rows_per_value = total_rows // num_unique_values

# Calculate the count of each ID
    id_counts = file['ID'].value_counts()

# Create a DataFrame to store the balanced data
    balanced_data = pd.DataFrame(columns=df.columns)

# Iterate over each unique value in column 'ID'
    for value in file['ID'].unique():
    # Extract rows with the current value
        subset = file[file['ID'] == value]
        # Calculate the number of rows for this value
        num_rows = len(subset)
        # If the number of rows is less than the target, duplicate rows to match the target
        if num_rows < target_rows_per_value:
            # Calculate the number of rows to duplicate
            rows_to_duplicate = target_rows_per_value - num_rows
            # Randomly select rows to duplicate
            rows_to_repeat = subset.sample(n=rows_to_duplicate, replace=True)
            # Append the repeated rows to the subset
            subset = pd.concat([subset, rows_to_repeat], ignore_index=True)
            # If the number of rows is more than the target, randomly select rows to match the target
        elif num_rows > target_rows_per_value:
        # Randomly select rows to keep
            subset = subset.sample(n=target_rows_per_value, replace=False)
    # Append the subset to the balanced data
        balanced_data = pd.concat([balanced_data, subset], ignore_index=True)
    return balanced_data


df=pd.read_csv("train_data.csv")
df=pd.DataFrame(df)
#df2=pd.read_csv("validation_data.csv")
#df2=pd.DataFrame(df2)
#df3=pd.read_csv("test_data.csv")
#df3=pd.DataFrame(df3)

new_data=datagenerator(df,10000)
new_data.to_csv('train_datab2.csv', index=False)

#new_data2=datagenerator(df2,10000)
#new_data2.to_csv('validation_datab.csv', index=False)

#new_data3=datagenerator(df3,6000)
#new_data3.to_csv('test_datab.csv', index=False)