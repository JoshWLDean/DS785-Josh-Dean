# -*- coding: utf-8 -*-

# LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree, export_text
import warnings

# %% IMPORT AND PREPROCESS DATA
# Ignore warnings and record start time
warnings.filterwarnings("ignore")
start_time = time.time()

# Load sales order data and perform initial cleanup
order_data = pd.read_csv('C:/Users/304742/OneDrive - Amcor/Documents/Data Science Capstone/Datasets/SalesOrderData.csv')
order_data['tech_spec_nbr'] = order_data['tech_spec_nbr'].fillna('Unknown')
order_data['salesOrderEnteredDate'] = pd.to_datetime(order_data['salesOrderEnteredDate'], format='%Y%m%d')

#Load backlog data and perform initial cleanup
backlog_data  = pd.read_csv('C:/Users/304742/OneDrive - Amcor/Documents/Data Science Capstone/Datasets/Backlog Size by Spec and Date.csv')
backlog_data['Backlog Snapshot Date'] = pd.to_datetime(backlog_data['Backlog Snapshot Date'])
backlog_data.iloc[:, 2:] = backlog_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0) #specs that don't use a workcenter will show 0 instead of NA

#Merge order data with backlog data
merged = order_data.merge(backlog_data, left_on='tech_spec_nbr', right_on='Tech Spec', how='left')
merged['date_diff'] = abs(merged['salesOrderEnteredDate'] - merged['Backlog Snapshot Date']).dt.days #dt.days turns the timedelta object into an integer
merged['date_diff'] = merged['date_diff'].fillna(9999) #using 9999 instead of NAs so that the idxmin() called next doesn't error
grouped_idx = merged.groupby(['salesOrderNbr', 'salesOrderLineNbr', 'tech_spec_nbr', 'salesOrderEnteredDate'])['date_diff'].idxmin() #find the "nearest" snapshot for each spec
data = merged.loc[grouped_idx] #filter down to just the "nearest" snapshots
data.filter(like="BacklogHours").fillna(0, inplace=True) #inserts zeroes for records that didn't find a match in the backlog data

# Manually specifying categorical columns and encoding them
categorical_columns = ['productionPlantNbr']
X = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
X = X.drop(columns=['salesOrderNbr',
                    'salesOrderLineNbr',
                    'originatingSalesOrderNbr',
                    'originatingSalesOrderLineNbr', 
                    'sourceMaterialNbr',
                    'sourceSystemNbr',
                    'deliveryPlantNbr',
                    'salesOrgId', 
                    'distributionChannel', 
                    'inventory_receipted_on_time',
                    'cmir_under_tolerance', #low importance
                    'date_diff',
                    'inventory_receipted_on_time'])
X = X.select_dtypes(include=['number', 'bool'])
y = data['on_time_flag']

# Impute missing values
imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# %% EDA
# Filter columns that do not contain 'backlog' or 'plant' in their names
filtered_columns = [col for col in X.columns if 'hours' not in col.lower() and 'plant' not in col.lower()]
filtered_X = X[filtered_columns]

# Calculate the correlation matrix for the filtered data including the on_time_flag
correlation_matrix = filtered_X.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
plt.title("Correlation Matrix of Independent Variables and Dependent Variable")
plt.show()

# List features for Feature Distribution Chart
features_to_show = ['LeadTime', 'sosl_schedulelinecount', 'sosl_minconfirmedMadDaysAway',
                    'po_prodordercount', 'Tier1_InputMaterialCount', 'Tier2_InputMaterialCount',
                    'ReqLeadTime', 'ConfvsReqLT', 'Tier1_AvgSchedStartDateDaysAway',
                    'ReqvsMatLeadTime', 'AvgSchedStartDatevsReqDate','on_time_flag']  

# Create a copy of X for visualization
X_copy = X.copy()

# Removing outliers (for visualization purposes) by filtering values outside the 1st and 99th percentiles for each feature
for feature in features_to_show:
    lower_quantile = X_copy[feature].quantile(0.01)
    upper_quantile = X_copy[feature].quantile(0.99)
    X_copy[feature] = X_copy[feature].clip(lower=lower_quantile, upper=upper_quantile)

# Plot distributions of the selected features
X_copy[features_to_show].hist(bins=20, figsize=(15, 10), color='lightblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# %% PREP DATA FOR MODEL TRAINING
# Remove on_time_flag from X
X = X.drop(columns=['on_time_flag'] )#,'ConfvsReqLT', 'AvgSchedStartDatevsReqDate', 'sosl_schedulelinecount', 'Tier1_AvgSchedStartDateDaysAway'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the list of models to test
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))
    #('Support Vector Machine', SVC(kernel='linear', random_state=42)),
    #('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5))
]

# Initialize SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# %% MODEL TRAINING

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'model__n_estimators': [100, 200 ,300],
        'model__max_depth': [None], #[None, 10, 20],
        'model__min_samples_split': [2,5] #[2, 5, 10]
    }
}

# Initialize empty variables to store results
results = []
final_model = None 
final_predictions = None
final_feat_imp = None
importance_df = None
best_f1_score = -np.inf  # Initialize the best F1 score as negative infinity

# Iterate through each model, applying hyperparameter tuning and cross-validation
for name, model in models:
    print(f"\nTuning hyperparameters for {name}...")
    
    # Wrap model in a pipeline to handle scaling if needed
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grids[name],
                               cv=cv,
                               scoring='f1',
                               n_jobs=-1,
                               verbose=1)
    
    # Conduct grid search
    grid_search.fit(X_train_smote, y_train_smote)
    
    # Retrieve the best estimator, parameters, and cross-validated score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        auroc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    except AttributeError:
        auroc = "N/A"

    # Append results with the best parameters and scores
    results.append({
        'Model': name,
        'Best CV F1 Score': round(best_score, 3),
        'Best Parameters': best_params,
        'Test Accuracy': round(accuracy, 3),
        'Test F1 Score': round(f1, 3),
        'Test AUROC': round(auroc, 3) if auroc != "N/A" else "N/A"
    })
    
    # Print feature importances if available
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_importances = best_model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        print(f"\nFeature Importances for {name}:\n", importance_df.sort_values(by='Importance', ascending=False))
        
    # Check if this model has the highest F1 score so far, and if so, store its results
    if f1 > best_f1_score:
        best_f1_score = f1  
        final_model = best_model  
        final_predictions = y_pred 
        final_feat_imp = importance_df
           
# Get predicted probabilities for the positive class
final_probs = final_model.predict_proba(X_test)[:, 1]

# %% REPORT RESULTS
# Convert results to DataFrame for easy displaya and print
results_df = pd.DataFrame(results)
print("\nModel Performance Summary with Hyperparameter Tuning:")
print(results_df)
print("\nBest Model Based on F1 Score:")
print(final_model)

# Record the end time and duration
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script took {elapsed_time:.2f} seconds to complete.")

# Frequency of orders being on time in the training data
total_orders = len(y_train)  # Total number of orders
on_time_count = y_train.value_counts().get(1, 0) 
late_count = y_train.value_counts().get(0, 0) 
on_time_frequency = (on_time_count / total_orders) * 100 
late_percentage = (late_count / total_orders) * 100  
print(f"\nTotal Orders: {total_orders}")
print(f"On-Time Orders: {on_time_count}")
print(f"Late Orders: {late_count}")
print(f"Percentage of Late Orders: {late_percentage:.2f}%")

# %% CONFUSION MATRIX
if final_model is not None and final_predictions is not None:
    # Generate confusion matrix
    cm = confusion_matrix(1-y_test, 1-final_predictions)

    # Convert the confusion matrix to a pandas DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=['True Class 0', 'True Class 1'], columns=['Predicted Class 0', 'Predicted Class 1'])

    # Display the confusion matrix as a table
    print(f"Confusion Matrix for {final_model.named_steps['model'].__class__.__name__} - Best Model:")
    print(cm_df)
else:
    print("No model or predictions available for confusion matrix.")

# %% ROC CURVE
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(1-y_test, 1-final_probs)

# Calculate the AUC score
auc_score = roc_auc_score(1-y_test,1- final_probs)

# Create list of classification thresholds to plot
threshold_values = [round(i * 0.1,1) for i in range(11)]

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line

# Fill the area under the curve
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.4, label=f'AUC = {auc_score:.2f}')

# Plot the points for the selected thresholds and add labels
for threshold in threshold_values:
    # Find the index of the closest threshold value
    threshold_index = (abs(thresholds - threshold)).argmin()
    
    # Get the corresponding FPR and TPR values for this threshold
    chosen_fpr = fpr[threshold_index]
    chosen_tpr = tpr[threshold_index]
    
    # Plot the threshold point with the appropriate color and label
    color = 'green' if threshold == 0.5 else 'blue'
    plt.scatter(chosen_fpr, chosen_tpr, color=color, zorder=5)
    plt.text(chosen_fpr, chosen_tpr, f'{threshold:.1f}', fontsize=9, color=color, ha='right', va='bottom')

# Add a label with the overall AUC score at specific point
plt.text(0.3, 0.7, f'AUC = {auc_score:.2f}', fontsize=12, color='black', ha='center', va='center', weight='bold')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
# %% OPTIMAL CLASSIFICATION THRESHOLD
# Find optimal threshold
# Threshold values to test
thresholds = np.arange(0.0, 1.0, 0.01)

# Empty list to store scores
f1_scores = []

# Loop through each threshold
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the threshold
    threshold_preds = (final_probs >= threshold).astype(int)
    
    # Calculate F1 score for the current threshold
    f1 = f1_score(1-y_test, 1-threshold_preds)
    f1_scores.append(f1)

# Find the threshold with the highest F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_score = max(f1_scores)

# Print 
print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1_score}")
# %%PLOT PREDICTED PROBABILITY DISTRIBUTIONS
# Create a df to hold predicted probability data
predicted_prob_df = pd.DataFrame({
    'Predicted Probability': 1-final_probs,
    'Actual': 1-y_test,
    'ConfirmedLate': (X_test['ConfvsReqLT'] >= 1).astype(int).tolist()
})

# Define custom colors for the classes
custom_colors = {0: 'Green', 1: 'Red'}

# Distribution of Predicted Probabilities
plt.figure(figsize=(10, 6))
sns.histplot(data=predicted_prob_df,
             x='Predicted Probability', hue='Actual', bins=30, kde=False, 
             edgecolor='black', multiple='stack', palette=custom_colors)
plt.xlabel('Predicted Probability of Lateness')
plt.ylabel('Count of Orders')
plt.legend(title='Actual Outcome', labels=['Late', 'On Time'])
plt.grid(True, alpha=0.3) 
plt.show()

# Distribution of Predicted Probabilities - Only those confirmed late
plt.figure(figsize=(10, 6))
sns.histplot(data=predicted_prob_df[predicted_prob_df['ConfirmedLate']==1], #Only those confirmed late
             x='Predicted Probability', hue='Actual', bins=30, kde=False, 
             edgecolor='black', multiple='stack', palette=custom_colors)
plt.xlabel('Predicted Probability of Lateness (Only Orders Confirmed Late)')
plt.ylabel('Count of Orders')
plt.legend(title='Actual Outcome', labels=['Late', 'On Time'])
plt.grid(True, alpha=0.3) 
plt.show()

# Distribution of Predicted Probabilities - Only those confirmed ontime
plt.figure(figsize=(10, 6))
sns.histplot(data=predicted_prob_df[predicted_prob_df['ConfirmedLate']==0], #Only those confirmed ontime
             x='Predicted Probability', hue='Actual', bins=30, kde=False, 
             edgecolor='black', multiple='stack', palette=custom_colors)
plt.xlabel('Predicted Probability of Lateness (Only Orders Confirmed On Time)')
plt.ylabel('Count of Orders')
plt.legend(title='Actual Outcome', labels=['Late', 'On Time'])
plt.grid(True, alpha=0.3) 
plt.show()

# Confirmed On Time/Late vs Actual Outcome
plt.figure(figsize=(10, 6))
sns.histplot(data=predicted_prob_df,
             x='ConfirmedLate', hue='Actual', bins=30, kde=False, 
             edgecolor='black', multiple='stack', palette=custom_colors)
plt.xlabel('Confirmed On Time (0) or Late (1)')
plt.ylabel('Count of Orders')
plt.legend(title='Actual Outcome', labels=['Late', 'On Time'])
plt.grid(True, alpha=0.3) 
plt.show()
# %% FEATURE IMPORTANCE
# Calculate the sum of importance for the production plant features
prodplant_feat_imps = final_feat_imp[final_feat_imp['Feature'].str.startswith('productionPlantNbr')]
avg_prodplant_imp = prodplant_feat_imps['Importance'].sum()
final_feat_imp_short = final_feat_imp[~final_feat_imp['Feature'].str.startswith('productionPlantNbr')]
final_feat_imp_short = pd.concat([final_feat_imp_short, pd.DataFrame({'Feature': ['productionPlantNbr_X (Sum)'], 'Importance': [avg_prodplant_imp]})], ignore_index=True)

# Sort the features by importance
final_feat_imp_sorted = final_feat_imp_short.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(6, 10))
sns.barplot(x='Importance', y='Feature', data=final_feat_imp_sorted, palette='viridis')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

# %% RANDOM FOREST RULE EXTRACTION TEST
# This section uses some code from a GitHub repo by GabeOwen at:
# https://github.com/GabeOw/Random-Forest-Rule-Extraction/blob/main/Random%20Forest%20Model%20Rules%20Extraction%20Code.ipynb

best_rf  = final_model.fit(X_train, y_train).named_steps['model']
# Initializing an empty list to store rules from decision trees
rules = []

# Iterating through each decision tree in the random forest
for tree in best_rf.estimators_:
    # Exporting the structure of the decision tree as text and appending it to the rules list
    r = export_text(tree, feature_names=list(X.columns), show_weights=True, max_depth=10)
    rules.append(r)

# Printing rules of the first 5 decision trees
for rule in rules[:5]:
    print(rule)
    
# Defining function to extract rules from a decision tree
def get_rules(tree, feature_names):
    tree_ = tree.tree_  # Getting the underlying tree structure
    # Creating a list of feature names, replacing undefined features with a placeholder string
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []  # List to store all the paths (rules) in the tree
    path = []  # Temporary list to store the current path

    # Recursive function to traverse the tree and collect rules
    def recurse(node, path, paths):
        # If the node is not a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 6)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 6)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            # If the node is a leaf node, append the current path to paths
            path += [(tree_.value[node], np.sum(tree_.value[node]))]
            paths += [path]

    # Starting the recursive function from the root node
    recurse(0, path, paths)

    # Sorting the paths based on the number of samples
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []  # List to store the rules in a formatted way
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        
        classes = path[-1][0][0]
        l = np.argmax(classes)
        class_label = l
        proba = np.round(100.0*classes[l]/np.sum(classes),2)
        samples = path[-1][1]
        
        # Appending the formatted rule to the rules list
        rules.append({
            'rule': rule,
            'class': class_label,
            'samples': samples,
            'proba': proba
        })
        
    return rules

# Getting feature names from the DataFrame df_train, excluding the target variable column
rules_feature_names = X_train.columns

# Creating an empty list to store the rules from each tree
rules_list = []

# Iterating through each decision tree in the random forest to collect rules
for tree in best_rf.estimators_:
    rules = get_rules(tree, rules_feature_names)
    rules_list.extend(rules)

# Converting the list of dictionaries into a DataFrame
rules_df = pd.DataFrame(rules_list)

# Renaming columns for better interpretation
rules_df.rename(columns={"rule": "Feature Rule"}, inplace=True)
rules_df = rules_df[rules_df["class"] == 1].copy()
rules_df.drop("class", axis=1, inplace=True)
rules_df.rename(columns={"samples": "Total Samples"}, inplace=True)
rules_df.rename(columns={"proba": "On Time Prob."}, inplace=True)
rules_df.reset_index(drop=True, inplace=True)

# Displaying the first 5 rows of the rules DataFrame
rules_df_head = rules_df.head()

# Print best rules
print('Feature Rules:', "\n")
pd.set_option('display.max_colwidth', None)
for item in rules_df['Feature Rule'][0:20]:
    print(item, "\n")
    