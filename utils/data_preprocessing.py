import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def merge_data(politic_data, economic_data, education_data, demographic_data, grade_rate_result_data):
    data = grade_rate_result_data.merge(politic_data, on=['Municipality', 'Year'], how='left')
    data = data.merge(economic_data, on=['Municipality', 'Year'], how='left')
    data = data.merge(education_data, on=['Municipality', 'Year'], how='left')
    data = data.merge(demographic_data, on=['Municipality', 'Year'], how='left')
    
    return data

def handle_missing_values(df):
    print(df.isnull().sum())
    df = df.dropna()  
    return df

def drop_highly_correlated_features(df, threshold=0.7):
    corr_matrix = df.corr().abs()
    print(corr_matrix)
    to_drop = set()
    for (row, col) in itertools.combinations(corr_matrix.columns, 2):
        if corr_matrix.loc[row, col] > threshold:
            if row != "Math_grade_rate_result" and col != "Math_grade_rate_result":
                if col not in to_drop:
                    print(f"Dropping {col} because of high correlation with {row} : {corr_matrix.loc[row, col]}")
                    to_drop.add(col)
    kept = [col for col in df.columns if col not in to_drop]
    
    return kept, list(to_drop)

def drop_highly_correlated_features_by_VIF(df, thresh=5.0):
    X = df.dropna()  
    X = X.select_dtypes(include=[np.number]).astype(float)  
    drop_features = [] 

    while True:
        X_const = sm.add_constant(X) 
        vif_data = pd.DataFrame()  
        vif_data["feature"] = X_const.columns 
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]  # محاسبه VIF
        print(vif_data)
        vif_no_const = vif_data[vif_data["feature"] != "const"] 
        max_vif_value = vif_no_const["VIF"].max()  

        if max_vif_value > thresh: 
            drop_feature = vif_no_const.sort_values("VIF", ascending=False)["feature"].iloc[0]  
            print(f"Dropping {drop_feature} with VIF = {max_vif_value:.2f}")
            X = X.drop(columns=[drop_feature])  
            drop_features.append(drop_feature) 
        else:
            break 
    return drop_features

def process_highly_correlated_features(df,exclude_features,corr_threshold=0.6, vif_threshold=5.0):

    keep_features, dropped_features = drop_highly_correlated_features(
        df[[col for col in df.columns if col not in exclude_features ]], corr_threshold)
    
    dropped_features.extend(drop_highly_correlated_features_by_VIF(
        df[[col for col in keep_features if col not in exclude_features]],vif_threshold))
    
    selected_features = [col for col in df.columns if col not in dropped_features]
    selected_features.extend(exclude_features)
    return selected_features, dropped_features,df[selected_features]
