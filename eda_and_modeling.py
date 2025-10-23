"""
Kaggle Credit Card Monthly Spending Prediction
Goal: Beat 250 RMSE and target 220 RMSE
Strategy: Advanced feature engineering + Gradient boosting ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT CARD SPENDING PREDICTION - KAGGLE COMPETITION")
print("=" * 80)

# Load data
print("\n[1/6] Loading data...")
train = pd.read_csv('analysis_data.csv')
test = pd.read_csv('scoring_data.csv')

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Basic info
print("\n" + "=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print("\nTraining Data Info:")
print(train.info())

print("\nMissing values:")
print(train.isnull().sum())

print("\nTarget variable (monthly_spend) statistics:")
print(train['monthly_spend'].describe())

print("\nCategorical variables:")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

print("\nNumerical variables:")
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'customer_id' in numerical_cols:
    numerical_cols.remove('customer_id')
if 'monthly_spend' in numerical_cols:
    numerical_cols.remove('monthly_spend')
print(numerical_cols)

# Distribution analysis
print("\n" + "=" * 80)
print("TARGET DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"Mean: ${train['monthly_spend'].mean():.2f}")
print(f"Median: ${train['monthly_spend'].median():.2f}")
print(f"Std: ${train['monthly_spend'].std():.2f}")
print(f"Min: ${train['monthly_spend'].min():.2f}")
print(f"Max: ${train['monthly_spend'].max():.2f}")
print(f"Skewness: {train['monthly_spend'].skew():.3f}")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION WITH TARGET (Top 15)")
print("=" * 80)

# Encode categorical variables for correlation analysis
train_encoded = train.copy()
le = LabelEncoder()
for col in categorical_cols:
    if col in train_encoded.columns:
        train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))

correlations = train_encoded.corr()['monthly_spend'].sort_values(ascending=False)
print(correlations.head(15))

# Check for obvious feature relationships
print("\n" + "=" * 80)
print("FEATURE ENGINEERING INSIGHTS")
print("=" * 80)

# Check if num_transactions * avg_transaction_value relates to monthly_spend
if 'num_transactions' in train.columns and 'avg_transaction_value' in train.columns:
    train['total_transaction_value'] = train['num_transactions'] * train['avg_transaction_value']
    corr = train[['monthly_spend', 'total_transaction_value', 'num_transactions', 'avg_transaction_value']].corr()
    print("\nTransaction relationship:")
    print(corr['monthly_spend'].sort_values(ascending=False))

# Income to credit ratio
if 'annual_income' in train.columns and 'credit_limit' in train.columns:
    train['income_to_credit_ratio'] = train['annual_income'] / (train['credit_limit'] + 1)
    print(f"\nIncome to credit ratio correlation: {train[['monthly_spend', 'income_to_credit_ratio']].corr().iloc[0,1]:.4f}")

# Credit utilization (if we can estimate it)
if 'credit_limit' in train.columns:
    train['credit_utilization'] = train['monthly_spend'] / (train['credit_limit'] + 1)
    print(f"Credit utilization stats:")
    print(train['credit_utilization'].describe())

print("\n" + "=" * 80)
print("CATEGORICAL VARIABLE ANALYSIS")
print("=" * 80)
for col in categorical_cols:
    if col != 'customer_id':
        print(f"\n{col}:")
        print(train.groupby(col)['monthly_spend'].agg(['mean', 'median', 'std', 'count']))

print("\n" + "=" * 80)
print("EDA COMPLETE - Ready for feature engineering and modeling!")
print("=" * 80)
