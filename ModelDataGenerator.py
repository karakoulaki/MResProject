#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:08:50 2024

@author: kk423
"""

import pandas as pd
import numpy as np
import random
import ast

def datagenerator(file,total_rows):

#calculate the number of different patterns 
    num_unique_values = len(file['ID'].unique())

# Calculate the number of times each pattern should be repeated
    target_rows_per_value = total_rows // num_unique_values

# Calculate the count of each pattern 
    id_counts = file['ID'].value_counts()

# Create a DataFrame to store the balanced data
    balanced_data = pd.DataFrame(columns=df.columns)

# Iterate over each pattern
    for value in file['ID'].unique():
        subset = file[file['ID'] == value]
        num_rows = len(subset)
        if num_rows < target_rows_per_value:
            rows_to_duplicate = target_rows_per_value - num_rows
            rows_to_repeat = subset.sample(n=rows_to_duplicate, replace=True)
            subset = pd.concat([subset, rows_to_repeat], ignore_index=True)
        elif num_rows > target_rows_per_value:
            subset = subset.sample(n=target_rows_per_value, replace=False)
        balanced_data = pd.concat([balanced_data, subset], ignore_index=True)
        balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

    return balanced_data

def datagenerator2(file, total_rows, extra_rows):
  
    def count_ones(matrix_str):
        matrix = ast.literal_eval(matrix_str)  # Convert string to list of lists
        return sum(sum(row) for row in matrix)  # Count the ones in the matrix
    
    # Apply the function to get the sum of ones for each matrix
    file['Ones_Count'] = file['Matrix'].apply(count_ones)
    
    # Identify the IDs with sum of matrix entries less than 3
    extra_ids = file[file['Ones_Count'] < 3]['ID'].unique()
        
    # Calculate the number of different patterns
    num_unique_values = len(file['ID'].unique())
    
    # Calculate the total extra rows
    total_extra_rows = len(extra_ids) * extra_rows
    
    # Ensure there are enough total rows to accommodate the extra rows
    if total_rows <= total_extra_rows:
        raise ValueError("The total number of rows is not enough to accommodate the extra rows for the specified IDs.")
    
    # Calculate the total rows minus the extra rows for the special IDs
    total_rows_normal = total_rows - total_extra_rows
    
    # Calculate the number of times each pattern should be repeated for the normal IDs
    num_normal_values = num_unique_values - len(extra_ids)
    if num_normal_values > 0:
        target_rows_per_value = total_rows_normal // num_normal_values
    else:
        target_rows_per_value = 0
    
    # Create a DataFrame to store the balanced data
    balanced_data = pd.DataFrame(columns=file.columns)
    
    # Iterate over each pattern
    for value in file['ID'].unique():
        subset = file[file['ID'] == value]
        num_rows = len(subset)
        
        # Determine the target rows for this ID
        if value in extra_ids:
            target_rows = target_rows_per_value + extra_rows
        else:
            target_rows = target_rows_per_value
        
        if num_rows < target_rows:
            rows_to_duplicate = target_rows - num_rows
            if rows_to_duplicate > 0:
                rows_to_repeat = subset.sample(n=rows_to_duplicate, replace=True)
                subset = pd.concat([subset, rows_to_repeat], ignore_index=True)
        elif num_rows > target_rows:
            subset = subset.sample(n=target_rows, replace=False)
        
        balanced_data = pd.concat([balanced_data, subset], ignore_index=True)
    
    # Shuffle the resulting DataFrame
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    
    return balanced_data


df=pd.read_csv("train_data_4.csv")
df=pd.DataFrame(df)
df2=pd.read_csv("validation_data.csv")
df2=pd.DataFrame(df2)
df3=pd.read_csv("test_data.csv")
df3=pd.DataFrame(df3)

new_data=datagenerator(df,100000)
new_data.to_csv('train_data4_more.csv', index=False)

#new_data2=datagenerator(df2,10000)
#new_data2.to_csv('validation_datab4.csv', index=False)

#new_data3=datagenerator(df3,6000)
#new_data3.to_csv('test_datab4.csv', index=False)