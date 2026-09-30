#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv(r"train_ctrUa4K.csv")
test_data = pd.read_csv(r"test_lAUu6dG.csv")


# ## ANALYSING

# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.shape


# In[6]:


test_data.shape


# In[7]:


train_data.info()


# In[8]:


train_data.describe()


# In[9]:


test_data.info()


# In[10]:


test_data.describe()


# In[11]:


train_data.isnull().sum()


# In[12]:


test_data.isnull().sum()


# In[13]:


print("Train duplicates:", train_data.duplicated().sum())
print("Test duplicates:", test_data.duplicated().sum())


# In[14]:


categorical_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Credit_History',
    'Property_Area',
    'Loan_Status'
]

for col in categorical_cols:
    print("\n", col)
    print(train_data[col].value_counts(dropna=False))


# In[15]:


numerical_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term'
]

train_data[numerical_cols].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[16]:


cols = ['Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed',
        'Credit_History', 'Property_Area',
        'Loan_Status']

for col in cols:
    train_data[col].value_counts(dropna=False).plot(
        kind='bar',
        title=col
    )
    plt.show()


# In[17]:


pd.crosstab(
    train_data['Credit_History'],
    train_data['Loan_Status'],
    margins=True
)


# In[18]:


pd.crosstab(
    train_data['Credit_History'],
    train_data['Loan_Status'],
    normalize='index'
) * 100


# In[19]:


train_data[
    train_data['Credit_History'].isna()
]['Loan_Status'].value_counts()


# In[20]:


train_data[
    train_data['Credit_History'].isna()
]['Loan_Status'].value_counts(normalize=True) * 100


# ## CLEANING

# In[21]:


train_cleaned = train_data.copy()
test_cleaned = test_data.copy()


# In[22]:


categorical_missing = [
    'Gender',
    'Married',
    'Dependents',
    'Self_Employed'
]

for col in categorical_missing:
    mode_value = train_cleaned[col].mode()[0]

    train_cleaned[col] = train_cleaned[col].fillna(mode_value)
    test_cleaned[col] = test_cleaned[col].fillna(mode_value)


# In[23]:


loan_median = train_cleaned['LoanAmount'].median()

train_cleaned['LoanAmount'] = train_cleaned['LoanAmount'].fillna(loan_median)
test_cleaned['LoanAmount'] = test_cleaned['LoanAmount'].fillna(loan_median)


# In[24]:


term_mode = train_cleaned['Loan_Amount_Term'].mode()[0]

train_cleaned['Loan_Amount_Term'] = train_cleaned['Loan_Amount_Term'].fillna(term_mode)
test_cleaned['Loan_Amount_Term'] = test_cleaned['Loan_Amount_Term'].fillna(term_mode)


# In[25]:


train_cleaned['Credit_History'] = train_cleaned['Credit_History'].fillna(2)
test_cleaned['Credit_History'] = test_cleaned['Credit_History'].fillna(2)


# In[26]:


print(train_cleaned.isnull().sum())
print(test_cleaned.isnull().sum())


# ## EDA

# In[27]:


train_cleaned['Loan_Status'].value_counts()


# In[28]:


train_cleaned['Loan_Status'].value_counts(normalize=True) * 100


# In[29]:


categorical_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Credit_History',
    'Property_Area'
]

for col in categorical_cols:
    print(f"\n--- {col} ---")

    result = pd.crosstab(
        train_cleaned[col],
        train_cleaned['Loan_Status'],
        normalize='index'
    ) * 100

    print(result.round(2))


# In[30]:


for col in categorical_cols:
    pd.crosstab(
        train_cleaned[col],
        train_cleaned['Loan_Status']
    ).plot(
        kind='bar',
        figsize=(6, 4)
    )

    plt.title(f'{col} vs Loan Status')
    plt.ylabel('Number of Applicants')
    plt.xticks(rotation=0)
    plt.show()


# In[31]:


train_cleaned.groupby('Loan_Status')[
    ['ApplicantIncome',
     'CoapplicantIncome',
     'LoanAmount',
     'Loan_Amount_Term']
].mean().round(2)


# In[32]:


train_cleaned.groupby('Loan_Status')[
    ['ApplicantIncome',
     'CoapplicantIncome',
     'LoanAmount',
     'Loan_Amount_Term']
].median()


# In[33]:


numerical_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term'
]

for col in numerical_cols:
    train_cleaned.boxplot(
        column=col,
        by='Loan_Status',
        figsize=(6, 4)
    )

    plt.title(f'{col} vs Loan Status')
    plt.suptitle('')
    plt.xlabel('Loan Status')
    plt.ylabel(col)
    plt.show()


# ## Feature Engineering

# In[34]:


train_cleaned['TotalIncome'] = (
    train_cleaned['ApplicantIncome'] +
    train_cleaned['CoapplicantIncome']
)

test_cleaned['TotalIncome'] = (
    test_cleaned['ApplicantIncome'] +
    test_cleaned['CoapplicantIncome']
)


# In[35]:


train_cleaned['LoanIncomeRatio'] = (
    train_cleaned['LoanAmount'] * 1000 /
    train_cleaned['TotalIncome']
)

test_cleaned['LoanIncomeRatio'] = (
    test_cleaned['LoanAmount'] * 1000 /
    test_cleaned['TotalIncome']
)


# In[36]:


train_cleaned[
    ['ApplicantIncome',
     'CoapplicantIncome',
     'TotalIncome',
     'LoanAmount',
     'LoanIncomeRatio']
].head()


# ## Model Preperation

# In[37]:


x = train_cleaned.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_cleaned['Loan_Status']


# In[38]:


x_test = test_cleaned.drop(columns=['Loan_ID'])


# In[39]:


print(x.shape)
print(y.shape)
print(x_test.shape)


# In[41]:


categorical_cols = x.select_dtypes(include='object').columns

categorical_cols


# In[42]:


x = pd.get_dummies(
    x,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

x_test = pd.get_dummies(
    x_test,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)


# In[43]:


x_test = x_test.reindex(
    columns=x.columns,
    fill_value=0
)


# In[44]:


print(x.shape)
print(x_test.shape)


# In[45]:


x.head()


# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


# In[48]:


print("X_train:", x_train.shape)
print("X_val:", x_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


# ## Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[50]:


lr_model = LogisticRegression(max_iter=1000)


# In[52]:


lr_model.fit(x_train, y_train)


# In[53]:


y_pred = lr_model.predict(x_val)


# In[55]:


accuracy = accuracy_score(y_val, y_pred)*100

print("Logistic Regression Accuracy:", accuracy)


# In[56]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_val, y_pred)
print(cm)


# In[57]:


print(classification_report(y_val, y_pred))


# ## Decision Tree

# In[59]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
    random_state=42
)
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_val)
dt_accuracy = accuracy_score(y_val, dt_pred) * 100
print("Decision Tree Accuracy:", dt_accuracy)


# In[60]:


print(confusion_matrix(y_val, dt_pred))
print(classification_report(y_val, dt_pred))


# ## Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_val)
rf_accuracy = accuracy_score(y_val, rf_pred) * 100
print("Random Forest Accuracy:", rf_accuracy)


# In[63]:


print(confusion_matrix(y_val, rf_pred))
print(classification_report(y_val, rf_pred))


# ## Cross-validation

# In[64]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# In[65]:


models = {
    'Logistic Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring='accuracy'
    )

    print(f"\n{name}")
    print("Fold Accuracies:", scores)
    print("Mean Accuracy:", scores.mean() * 100)
    print("Std:", scores.std() * 100)


# ## Machine Learning – Final Model Development
# 
# The previous data cleaning and EDA steps were performed to understand the dataset, identify missing values, study distributions, detect unusual values, and analyze the relationships between applicant characteristics and loan approval.
# 
# For the final machine learning process, the original training and test datasets are used again instead of the manually cleaned datasets.
# 
# This is done to prevent **data leakage**. Preprocessing operations such as missing-value imputation and categorical encoding should be learned only from the training data and then applied to validation/test data. Therefore, these preprocessing steps will be included inside a machine learning pipeline.
# 
# The final ML workflow will be:
# 
# 1. Start with the original train and test datasets.
# 2. Perform feature engineering.
# 3. Separate input features (X) and target (y).
# 4. Split the training data into training and validation sets.
# 5. Build a preprocessing pipeline for missing values and categorical encoding.
# 6. Train and compare classification models.
# 7. Evaluate models using validation accuracy and cross-validation.
# 8. Select the final model.
# 9. Train the selected pipeline using the complete training dataset.
# 10. Predict `Loan_Status` for the provided test dataset.
# 
# This ensures that model evaluation is performed using a clean and consistent machine learning workflow without allowing validation or test information to influence model training.

# In[66]:


train_ml = train_data.copy()
test_ml = test_data.copy()


# In[67]:


train_ml['TotalIncome'] = (
    train_ml['ApplicantIncome'] +
    train_ml['CoapplicantIncome']
)

test_ml['TotalIncome'] = (
    test_ml['ApplicantIncome'] +
    test_ml['CoapplicantIncome']
)


# In[68]:


train_ml['LoanIncomeRatio'] = (
    train_ml['LoanAmount'] * 1000 /
    train_ml['TotalIncome']
)

test_ml['LoanIncomeRatio'] = (
    test_ml['LoanAmount'] * 1000 /
    test_ml['TotalIncome']
)


# In[69]:


train_ml[
    ['ApplicantIncome',
     'CoapplicantIncome',
     'TotalIncome',
     'LoanAmount',
     'LoanIncomeRatio']
].head()


# In[70]:


x = train_ml.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_ml['Loan_Status']


# In[71]:


x_test = test_ml.drop(columns=['Loan_ID'])


# In[72]:


print("X shape:", x.shape)
print("y shape:", y.shape)
print("X_test shape:", x_test.shape)


# In[73]:


x.info()


# In[74]:


categorical_features = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area'
]


# In[75]:


credit_feature = ['Credit_History']


# In[76]:


numerical_features = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'TotalIncome',
    'LoanIncomeRatio'
]


# In[77]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# In[78]:


categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore'
        ))
    ]
)


# In[79]:


credit_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(
            strategy='constant',
            fill_value=-1
        )),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore'
        ))
    ]
)


# In[80]:


numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ]
)


# In[81]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('credit', credit_transformer, credit_feature),
        ('num', numerical_transformer, numerical_features)
    ]
)


# In[83]:


x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


# In[84]:


print("X_train:", x_train.shape)
print("X_val:", x_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


# ## LOGISTIC REGRESSION

# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[86]:


lr_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ]
)


# In[87]:


lr_pipeline.fit(x_train, y_train)


# In[88]:


lr_pred = lr_pipeline.predict(x_val)


# In[89]:


lr_accuracy = accuracy_score(y_val, lr_pred) * 100

print("Logistic Regression Accuracy:", lr_accuracy)


# In[90]:


print(confusion_matrix(y_val, lr_pred))
print(classification_report(y_val, lr_pred))


# ## DECISION TREE

# In[91]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[92]:


dt_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(random_state=42))
    ]
)


# In[94]:


dt_pipeline.fit(x_train, y_train)


# In[95]:


dt_pred = dt_pipeline.predict(x_val)

dt_accuracy = accuracy_score(y_val, dt_pred) * 100

print("Decision Tree Accuracy:", dt_accuracy)


# ## RANDOM FOREST

# In[96]:


rf_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# In[97]:


rf_pipeline.fit(x_train, y_train)


# In[99]:


rf_pred = rf_pipeline.predict(x_val)

rf_accuracy = accuracy_score(y_val, rf_pred) * 100

print("Random Forest Accuracy:", rf_accuracy)


# ## Gradient Boosting

# In[100]:


from sklearn.ensemble import GradientBoostingClassifier


# In[101]:


gb_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(
            random_state=42
        ))
    ]
)


# In[102]:


gb_pipeline.fit(x_train, y_train)


# In[104]:


gb_pred = gb_pipeline.predict(x_val)


# In[105]:


gb_accuracy = accuracy_score(y_val, gb_pred) * 100

print("Gradient Boosting Accuracy:", gb_accuracy)


# ## 5-Fold Cross-Validation

# In[106]:


from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[107]:


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# In[108]:


models = {
    'Logistic Regression': lr_pipeline,
    'Decision Tree': dt_pipeline,
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gb_pipeline
}


# In[110]:


for name, model in models.items():

    scores = cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring='accuracy'
    )

    print(f"\n{name}")
    print("Fold Accuracies:", scores)
    print("Mean Accuracy:", scores.mean() * 100)
    print("Standard Deviation:", scores.std() * 100)


# ## FINAL PIPELINE

# In[111]:


final_model = lr_pipeline

final_model.fit(x, y)


# In[112]:


test_predictions = final_model.predict(x_test)


# In[113]:


print(test_predictions.shape)
print(test_predictions[:20])


# In[114]:


print(pd.Series(test_predictions).value_counts())


# In[115]:


submission = pd.DataFrame({
    'Loan_ID': test_data['Loan_ID'],
    'Loan_Status': test_predictions
})


# In[116]:


print(submission.head(10))
print("\nShape:", submission.shape)
print("\nMissing values:")
print(submission.isnull().sum())


# In[117]:


print(submission['Loan_Status'].unique())


# In[121]:


submission.to_csv(r'loan_prediction_submission.csv', index=False)


# In[ ]:




