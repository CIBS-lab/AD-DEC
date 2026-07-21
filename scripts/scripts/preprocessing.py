#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:21:57 2024

@author: cocotirambulo
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from sklearn.preprocessing import LabelEncoder


#### LOAD DATA SET
f_wrap_data_cont="/groups/rbrinton/coco/wrap/data/df_model2_bincat_age.xlsx"
f_wrap_data_catbin=r"/groups/rbrinton/coco/wrap/data/df_model2_bincat_age.xlsx"

#Results 
output_directory = "/Users/cocotirambulo/Library/CloudStorage/Box-Box/RUSH/WRAP/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types-main/data"  
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#Read excel files 
wrap_info_cont=pd.read_excel(f_wrap_data_cont, dtype={'WRAPNo': str})
wrap_info_catbin=pd.read_excel(f_wrap_data_catbin, dtype={'WRAPNo': str})

# =============================================================================
# CALCULATE COMORBIDITIES
# =============================================================================

#Fix comorbidities 
# Create a new column 'final_bmi_cat'
wrap_info_catbin['final_bmi_cat'] = wrap_info_catbin.groupby('WRAPNo')['bmi_obesity_cat'].transform(lambda x: 1 if any(x == 1) else 0)
wrap_info_catbin['final_bmi_cat'] = wrap_info_catbin.groupby('WRAPNo')['final_bmi_cat'].transform(lambda x: 0 if all(x == 0) else x)
wrap_info_catbin = wrap_info_catbin[['WRAPNo', 'bmi_obesity_cat', 'final_bmi_cat'] + [col for col in wrap_info_catbin.columns if col not in ['WRAPNo', 'bmi_obesity_cat', 'final_bmi_cat']]]

# Create a new column 'final_htn_cat'
wrap_info_catbin['final_htn_cat'] = wrap_info_catbin.groupby('WRAPNo')['Hyperten_E_bin'].transform(lambda x: 1 if any(x == 1) else 0)

# Create a new column 'final_diab_cat'
wrap_info_catbin['final_diab_cat'] = wrap_info_catbin.groupby('WRAPNo')['Diab_E_bin'].transform(lambda x: 1 if any(x == 1) else 0)

# Create a new column 'final_highchol_cat'
wrap_info_catbin['final_highchol_cat'] = wrap_info_catbin.groupby('WRAPNo')['HighChol_E_bin'].transform(lambda x: 1 if any(x == 1) else 0)

# Create a new column 'final_consensus_dx'
wrap_info_catbin['final_consensus_dx'] = wrap_info_catbin.groupby('WRAPNo')['consensus_dx_bin'].transform(lambda x: 1 if any(x == 1) else 0)

# Reorder columns
column_order = ['WRAPNo','VisNo','AgeAtVisit','gender','carrier','race','consensus_dx_bin', 'final_consensus_dx','bmi_obesity_cat', 'final_bmi_cat', 'Hyperten_E_bin', 'final_htn_cat', 'Diab_E_bin', 'final_diab_cat', 'HighChol_E_bin', 'final_highchol_cat']
column_order2 = ['WRAPNo','VisNo','AgeAtVisit','gender','carrier','race', 'final_bmi_cat', 'final_htn_cat', 'final_diab_cat', 'final_highchol_cat']
column_order2cog = ['WRAPNo','VisNo','AgeAtVisit','gender','carrier','race','final_consensus_dx', 'final_bmi_cat', 'final_htn_cat', 'final_diab_cat', 'final_highchol_cat']

wrap_info_catbin_all = wrap_info_catbin[column_order]
wrap_info_catbin = wrap_info_catbin[column_order2]
wrap_info_catbin_cog=wrap_info_catbin_all[column_order2cog]

wrap_info_catbin_first_ageatvisit = wrap_info_catbin.groupby('WRAPNo').first().reset_index()
wrap_info_catbin_first_ageatvisit_cog = wrap_info_catbin_cog.groupby('WRAPNo').first().reset_index()
wrap_info_catbin_first_ageatvisit_cog_final = wrap_info_catbin_first_ageatvisit_cog.iloc[:, 2:]
wrap_info_catbin_cog_final = wrap_info_catbin_first_ageatvisit_cog.iloc[:, 3:]

# Select columns containing variables (excluding the 'WRAPNo' and 'VisNo' columns)
variables_cont = wrap_info_cont.columns[2:]

df_time_series_cont = pd.DataFrame(wrap_info_cont)
df_time_series_cont_nonnorm = pd.DataFrame(wrap_info_cont)

# =============================================================================
# CALCULATE FEATURES: MEAN AND VARIANCE
# =============================================================================

#### CALCULATE THE WEIGHTED MEAN, VARIANCE, SKEWNESS, AND KURTOSIS OF EACH VARIABLE ####

df_time_series_cont_nonnorm = pd.DataFrame(df_time_series_cont_nonnorm)

def calculate_statistics(group):
    # Extract the weights from the 'VisNo' column of the group
    weights = group['VisNo']
    
    # Extract the data columns (excluding the identifier column WRAPNo and VisNo)
    data_columns = group.iloc[:, 2:]
    
    # Initialize an empty list to store the results
    result_list = []
    
    # Calculate and append the statistics for each variable
    for col in data_columns.columns:
        # Calculate the weighted mean for the variable
        weighted_mean = (group[col].values * weights.values).sum() / weights.sum()
        result_list.append(weighted_mean)
        
        # Calculate the variance for the variable (without weighting)
        variance = np.var(group[col])
        result_list.append(variance)
        
        # # Calculate the skewness for the variable (without weighting)
        # skew_val = skew(group[col])
        # result_list.append(skew_val)
        
        # # Calculate the kurtosis for the variable (without weighting)
        # kurt_val = kurtosis(group[col])
        # result_list.append(kurt_val)
    
    # Append "_wm", "_var", "_skew", and "_kurt" to each variable's original column name
    columns_result = []
    for col in data_columns.columns:
        columns_result.append(f"{col}_wm")
        columns_result.append(f"{col}_var")
        # columns_result.append(f"{col}_skew")
        # columns_result.append(f"{col}_kurt")
    
    # Create a new DataFrame with calculated statistics
    result = pd.DataFrame([result_list], columns=columns_result)
    
    return result

# Apply the calculate_statistics function grouped by WRAPNo
statistics_df = df_time_series_cont_nonnorm.groupby('WRAPNo').apply(calculate_statistics).reset_index(drop=True)

# Fill NaN values with 0
statistics_df_nona = statistics_df.fillna(0)

# Add the first column from df1 to df2
statistics_df_nona['AgeAtFirstVisit'] = wrap_info_catbin_first_ageatvisit_cog_final.iloc[:, 0]

col2drop = [c for c in statistics_df_nona.columns if not ('pacc4' in c or 'pTau217_cv' in c)]
statistics_df_nona_1 = statistics_df_nona[col2drop]

# =============================================================================
# CALCULATE VARIANCE INFLUENCE FACTOR
# =============================================================================

def calculate_vif_(X, thresh=5.0):
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables[:-1]])
    return X.iloc[:, variables[:-1]]

# Create a StandardScaler object
scaler = StandardScaler()
statistics_df_nona_z=statistics_df_nona_1
# Extract numerical columns from the DataFrame
numerical_columns = statistics_df_nona_z.select_dtypes(include=['float64', 'int64']).columns
# Fit the scaler on the numerical data and transform it
statistics_df_nona_z[numerical_columns] = scaler.fit_transform(statistics_df_nona_z[numerical_columns])

all_cols_wmeans = [c for c in statistics_df_nona_z.columns if 'wm' in c or 'AgeAtFirstVisit' in c and 'var' not in c]
table_data_wm = statistics_df_nona_z[all_cols_wmeans]

prova=calculate_vif_(table_data_wm)
#drop Non_HDL_Chol and choles1

col2drop2 = [c for c in table_data_wm.columns if not ('Non_HDL_Chol' in c or 'choles1' in c)]
table_data_continous_final = statistics_df_nona_z[col2drop2]

file_z_continuous = f"/Users/cocotirambulo/Library/CloudStorage/Box-Box/RUSH/WRAP/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types-main/x_dec/data/data_z_continuous.xlsx"
# Save the DataFrame to an Excel file
table_data_continous_final.to_excel(file_z_continuous, index=False)

# =============================================================================
# Label encoding of categorical variables using the built-in function LabelEncoder()
# =============================================================================

wrap_info_catbin.head()
labelled_wrap_catbin=wrap_info_catbin

labelled_wrap_catbin = labelled_wrap_catbin.drop_duplicates(subset=['WRAPNo']).set_index('WRAPNo')


# Ready to encode categorical variables (encode names with values)
le = LabelEncoder()

labelled_wrap_catbin['gender']= le.fit_transform(labelled_wrap_catbin['gender'])
labelled_wrap_catbin['carrier']= le.fit_transform(labelled_wrap_catbin['carrier'])
labelled_wrap_catbin['race']= le.fit_transform(labelled_wrap_catbin['race'])
labelled_wrap_catbin['final_bmi_cat']= le.fit_transform(labelled_wrap_catbin['final_bmi_cat'])
labelled_wrap_catbin['final_htn_cat']= le.fit_transform(labelled_wrap_catbin['final_htn_cat'])
labelled_wrap_catbin['final_diab_cat']= le.fit_transform(labelled_wrap_catbin['final_diab_cat'])
labelled_wrap_catbin['final_highchol_cat']= le.fit_transform(labelled_wrap_catbin['final_highchol_cat'])

labelled_wrap_catbin_final = labelled_wrap_catbin.drop(columns=['VisNo','AgeAtVisit'])

file_labelled_cat = f"/Users/cocotirambulo/Library/CloudStorage/Box-Box/RUSH/WRAP/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types-main/x_dec/data/data_labelled_cat.xlsx"
# Save the DataFrame to an Excel file
labelled_wrap_catbin_final.to_excel(file_labelled_cat, index=False)

# =============================================================================
# Encoding of categorical variables using pandas
# =============================================================================

# Create a copy of the DataFrame
wrap_info_catbin2 = wrap_info_catbin.copy()

# Change Female == 2 to Female == 0
wrap_info_catbin2['gender'] = wrap_info_catbin2['gender'].replace(2, 0)

# Convert categorical variables to categorical data type
cat_vars = ['race']
wrap_info_catbin2[cat_vars] = wrap_info_catbin2[cat_vars].astype("category")

# Create dummy variables for 'race' column
dummy_race = pd.get_dummies(wrap_info_catbin2['race'], prefix='race')

# Remove the 'race' column from wrap_info_catbin2
wrap_info_catbin2.drop(columns=['race'], inplace=True)

# Concatenate wrap_info_catbin2 with dummy_race
wrap_info_catbin2 = pd.concat([wrap_info_catbin2, dummy_race], axis=1)

# Extract and store the "WRAPNo" column
wrap_no_column = wrap_info_catbin2["WRAPNo"]

# Drop the "WRAPNo" column from the DataFrame
wrap_info_catbin2.drop(columns=["WRAPNo"], inplace=True)

# Get dummies for all remaining categorical columns
pandas_labelled_wrap_catbin_all = pd.get_dummies(wrap_info_catbin2, drop_first=True)

# Concatenate "WRAPNo" column back to the DataFrame
pandas_labelled_wrap_catbin_all["WRAPNo"] = wrap_no_column

# Convert True/False to 1/0
pandas_labelled_wrap_catbin_final = pandas_labelled_wrap_catbin_all.applymap(lambda x: 1 if x == True else (0 if x == False else x))
pandas_labelled_wrap_catbin_final = pandas_labelled_wrap_catbin_final.drop_duplicates(subset=['WRAPNo']).set_index('WRAPNo')
pandas_labelled_wrap_catbin_final = pandas_labelled_wrap_catbin_final.drop(columns=['VisNo','AgeAtVisit'])

file_pandas_labelled_cat = f"/Users/cocotirambulo/Library/CloudStorage/Box-Box/RUSH/WRAP/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types-main/x_dec/data/data_pandas_labelled_cat.xlsx"
# Save the DataFrame to an Excel file
pandas_labelled_wrap_catbin_final.to_excel(file_pandas_labelled_cat, index=False)

# =============================================================================
# RACE PROBABILITY DISTRIBUTION FUNCTION 
# =============================================================================

# # Create a new DataFrame df_race with unique 'WRAPNo' values as the index
# df_race = pd.DataFrame(wrap_info_catbin['race'].astype(str).copy())
# df_race['WRAPNo'] = wrap_info_catbin['WRAPNo']
# df_race = df_race.drop_duplicates(subset=['WRAPNo']).set_index('WRAPNo')

# # print race frequency distribution
# print(df_race.value_counts())
# # print race distribution by percentage (probability)
# print(df_race.value_counts()/len(df_race))

# # Create categorical plot
# #g = sns.catplot(data=df_race, x="race", kind="count");
# plot=sns.displot(df_race, x="race", shrink=.8);

# # Given distribution in the sample of 389 patients
# observed_races = ['White', 'Black', 'Hispanic', 'Asian']
# observed_frequencies = [374, 11, 3, 1]

# # Calculate probabilities from observed frequencies
# total_patients = sum(observed_frequencies)
# observed_probabilities = [freq / total_patients for freq in observed_frequencies]

# # Create a weighted probability distribution function
# def weighted_race_pdf(race):
#     try:
#         return observed_probabilities[observed_races.index(race)]
#     except ValueError:
#         return 0

# # Generate a set of values for the races
# race_values = np.linspace(0, len(observed_races) - 1, 1000)

# # Calculate the PDF values for each race
# pdf_values = [weighted_race_pdf(race) for race in observed_races]

# # Plot the PDF
# plt.bar(observed_races, pdf_values, color=['blue', 'orange', 'green', 'red'])
# plt.title('Weighted Probability Distribution Function of Races')
# plt.xlabel('Race')
# plt.ylabel('Probability')
# plt.show()

# # Create a mapping between race values and pdf_values
# race_mapping = {4: 0.961440, 3: 0.028278, 5: 0.007712, 2: 0.002571}

# # Map the pdf_values to the 'race' column to create the 'race_prob' variable
# wrap_info_catbin['race_prob'] = wrap_info_catbin['race'].map(race_mapping)


# # Display the DataFrame with the new variables
# print(wrap_info_catbin)

# # Assuming 'race' column exists in wrap_info_catbin
# wrap_info_catbin_final = wrap_info_catbin.drop(['AgeAtVisit','VisNo', 'race'], axis=1)

# wrap_info_catbin_final1 = wrap_info_catbin_final.drop_duplicates(subset=['WRAPNo']).set_index('WRAPNo')


# file_z_cat = f"/Users/cocotirambulo/Library/CloudStorage/Box-Box/RUSH/WRAP/Deep-Embedded-Clustering-generalisability-and-adaptation-for-mixed-data-types-main/x_dec/data/data_z_categorical.xlsx"
# # Save the DataFrame to an Excel file
# wrap_info_catbin_final1.to_excel(file_z_cat, index=False)


