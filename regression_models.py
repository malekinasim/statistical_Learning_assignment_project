import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from linearmodels.panel import RandomEffects, PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def model_diagnostics(model, X, plot_name):
    # Residuals
    try:
        residuals = model.resid
        fitted_values = model.fittedvalues
    except AttributeError:
        residuals = model.resids
        fitted_values = model.fitted_values
    
    # Residuals distribution
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.savefig(f"results/plot/{plot_name}_residuals_distribution.png", dpi=500) 
    plt.close()

    # Residuals vs Fitted Values plot
    plt.scatter(fitted_values, residuals)
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig(f"results/plot/{plot_name}_Residuals vs Fitted Values.png", dpi=500) 
    plt.close()
        
    # Residuals vs leverage
    if hasattr(model, "get_influence"): 
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
        plt.title("Residuals vs Leverage")
        plt.savefig(f"data/{plot_name}_Residuals vs leverage.png", dpi=500)
        plt.close()
    else:   
        plt.scatter(fitted_values, residuals)
        plt.title("Residuals vs Fitted Values (Panel Model)")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.axhline(0, color='red', linestyle='--')
        plt.savefig(f"results/plot/{plot_name}_residuals_vs_fitted_panel.png", dpi=500)
        plt.close()
        
    
    
    
    return residuals, fitted_values


def OLS_full_model(X, y):
    # Add constant to the data
    X = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X).fit()    
    return model
def OLS_backward_elimination(X, y, significance_level=0.25):
    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()
    while True:
        pvalues = model.pvalues.drop("const", errors='ignore')
        max_p_value = pvalues.max()
        if max_p_value < significance_level:
            break
        
        excluded_feature = pvalues.idxmax()
        X_temp= X.drop(columns=[excluded_feature])
        new_model = sm.OLS(y, X_temp).fit()
        if(new_model.rsquared_adj >= model.rsquared_adj):
            print(f"Dropping {excluded_feature} with p-value = {max_p_value:.4f} because Adj R²: {model.rsquared_adj:.4f} → {new_model.rsquared_adj:.4f}")
            model = new_model
            X= X_temp
        else:
            print(f"Keeping {excluded_feature} with p-value = {max_p_value:.4f} because Adj R²: {model.rsquared_adj:.4f}")
            break
    return model
def fixed_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
    df = df.set_index([entity, time])
    observation_size = len(df)  
    remaining_features = features.copy()
    prev_adj_r2 = -np.inf  
    while True:
        formula = f"{dep_var} ~ {' + '.join(remaining_features)} + EntityEffects + TimeEffects"

        model = PanelOLS.from_formula(formula, data=df).fit()
        pvalues = model.pvalues[remaining_features]
        max_pval = pvalues.max()
        excluded_feature = pvalues.idxmax()

        if max_pval < significance_level:
            break
        

        parameter_size= len(model.params) 
        adj_r2 = 1 - ((1 - model.rsquared) * ( observation_size- 1)) / (observation_size - (parameter_size+1))
        if adj_r2>= prev_adj_r2:
            remaining_features.remove(excluded_feature)
            print(f"Dropping {excluded_feature} with p = {max_pval:.4f} because Adj R² {prev_adj_r2:.4f}-> {adj_r2:.4f}")
            prev_adj_r2 = adj_r2
        else:
            print(f"Keeping {excluded_feature} with p = {max_pval:.4f} because  Adj R² decreased stopped")
            break
    
    final_formula = f"{dep_var} ~ {' + '.join(remaining_features)} + EntityEffects + TimeEffects"
    final_model = PanelOLS.from_formula(final_formula, data=df).fit()
    
    return final_model
def random_effects_panel_backward_elimination(df, dep_var, features, entity, time, significance_level=0.25):
    df = df.set_index([entity, time])
    observation_size = len(df)
    num_entities = df.index.get_level_values(entity).nunique()
    remaining_features = features.copy()
    prev_adj_r2 = -np.inf

    while True:

        if len(remaining_features) >= num_entities:
            print("Stopping: Too many predictors for available entities.")
            break

        formula = f"{dep_var} ~ {' + '.join(remaining_features)}"
        try:
            model = RandomEffects.from_formula(formula, data=df).fit()
        except ZeroDivisionError as e:
            print("Model fitting failed due to ZeroDivisionError:", e)
            break

        pvalues = model.pvalues[remaining_features]
        max_pval = pvalues.max()
        excluded_feature = pvalues.idxmax()

        if max_pval < significance_level:
            print("All remaining features are significant.")
            break

        parameter_size = len(model.params)
        adj_r2 = 1 - ((1 - model.rsquared) * (observation_size - 1)) / (observation_size - (parameter_size + 1))

        if adj_r2 >= prev_adj_r2:
            remaining_features.remove(excluded_feature)
            print(f"Dropping {excluded_feature} with p = {max_pval:.4f} because Adj R² {prev_adj_r2:.4f} -> {adj_r2:.4f}")
            prev_adj_r2 = adj_r2
        else:
            print(f"Keeping {excluded_feature} with p = {max_pval:.4f} because Adj R² decreased. Stopping.")
            break

    
    if remaining_features:
        final_formula = f"{dep_var} ~ {' + '.join(remaining_features)}"
        final_model = RandomEffects.from_formula(final_formula, data=df).fit()
        return final_model
    else:
        print("No features left to build a final model.")
        return None

import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(model, X, y, plot_name):
    try:
     fitted_values = model.fittedvalues
    except AttributeError:
       fitted_values = model.fitted_values

    plt.figure(figsize=(8, 6))
    plt.scatter(y, fitted_values, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Fit')  # خط مناسب
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f"results/plot/{plot_name}_actual_vs_predicted.png", dpi=500)
    plt.show()
    






