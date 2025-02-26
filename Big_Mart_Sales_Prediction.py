#import required frameworks
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

df11=pd.read_csv("train_v9rqX0R.csv")
df22=pd.read_csv("test_AbJTz2l.csv")

df1=df11
df2=df22

df1.head()
df2.head()

df1.isnull().sum()
df2.isnull().sum()

df1['Item_Weight'].value_counts().sum()

df1['Outlet_Size'].nunique()

df2['Outlet_Size'].value_counts().sum()

#filling the missing value
def missing_value(df):
    df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace=True)
    df["Outlet_Size"].fillna(df["Outlet_Size"].mode()[0], inplace=True)
    return df

df1 = missing_value(df1)

df2 = missing_value(df2)

df1.duplicated().sum()

df2.duplicated().sum()

#check the outliers of numerical features
numeric_columns = df1.select_dtypes(include=['int64', 'float64']).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 6))  # Set the figure size for each plot
    sns.boxplot(x=df1[column])
    plt.title(f'Boxplot for {column}')
    plt.show()  # Show the plot for each column

numeric_columns = df2.select_dtypes(include=['int64', 'float64']).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 6))  # Set the figure size for each plot
    sns.boxplot(x=df1[column])
    plt.title(f'Boxplot for {column}')
    plt.show()  # Show the plot for each column

def cap_outliers(df, column):
    # Calculate Q1, Q3, and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the outliers
    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else x)
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else x)

    return df

df1 = cap_outliers(df1, "Item_Visibility")
df2 = cap_outliers(df2, "Item_Visibility")

df1.select_dtypes(include = ["object"]).nunique()

unique_values = {col:df1[col].unique() for col in df1.select_dtypes(include = ["object"]).columns}

for col, values in unique_values.items():
    print(f"Unique values in '{col}': {values}")

def regular_value(df):
    df["Item_Fat_Content"].replace({'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}, inplace=True)
    return df

df1 = regular_value(df1)
df2 = regular_value(df2)

unique_values = {col:df1[col].unique() for col in df1.select_dtypes(include = ["object"]).columns}
for col, values in unique_values.items():
    print(f"Unique values in '{col}': {values}")

categorical_columns = df1.select_dtypes(include=['object']).columns.tolist()
categorical_columns

# label encoding categarical columns
Le = LabelEncoder()
for col in categorical_columns:
    df1[col] = Le.fit_transform(df1[col]) 
    df2[col] = Le.fit_transform(df2[col])

#Splitting the model for training and testing
x = df1.drop('Item_Outlet_Sales', axis=1, inplace=False)
y = df1['Item_Outlet_Sales']

x_train,x_val,y_train,y_val = train_test_split(x, y, train_size=0.2, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(df2)

# Fitting various ML models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    results[name] = {'RMSE': rmse, 'R2 Score': r2}

for model, metrics in results.items():
    print(f'{model}: RMSE={metrics["RMSE"]}, R2 Score={metrics["R2 Score"]}')

best_model = min(results, key=lambda k: results[k]['RMSE'])
print(f'Best model: {best_model}')

#Hyperparameter tuning
param_grid = {
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'GradientBoosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]},
    'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]},
}

#selecting bestmodel
if best_model in param_grid:
    grid_search = GridSearchCV(models[best_model], param_grid[best_model], cv=5, n_jobs=-1)
    grid_search.fit(x_train_scaled, y_train)
    best_params = grid_search.best_params_
    print(f'Best Parameters for {best_model}: {best_params}')
    best_estimator = grid_search.best_estimator_
else:
    best_estimator = models[best_model]

# Train best model with best parameters
best_estimator.fit(x_train_scaled, y_train)
y_val_pred = best_estimator.predict(x_val_scaled)

# Evaluate best model
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
print(f'Final Best Model Performance - RMSE: {rmse}, R2 Score: {r2}')

# Final predictions on test data
test_predictions = best_estimator.predict(x_test_scaled)
df_test=pd.read_csv("test_AbJTz2l.csv")
df_test['Item_Outlet_Sales'] = test_predictions
df_test[['Item_Identifier','Outlet_Identifier', 'Item_Outlet_Sales']].to_csv("submission.csv", index=False)

print("Pipeline Complete! Best model used for predictions and results saved in submission.csv")

