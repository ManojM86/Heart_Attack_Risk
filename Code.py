# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# Data Preprocessing

# Reading a CSV file into a DataFrame(df)

df = pd.read_csv("heart_attack_prediction_dataset.csv")

#df
# Correlation between all the Variables"""

#plt.figure(figsize=(20,10))
#sns.heatmap(df.corr(),annot=True)

#Among all the Independent Variables Age and Smoke is having relatively high positive correlation coefficient whereas other Independent variables have very low correlation coefficient.
# Converting the Blood pressure column into High, Low, Normal categories

#### Conerting the Blood Pressure column directly to cateogorical columns by label encoding will not give an accurate result as there is a repition in the values. Coverting into High, Low and Normal will give the better results

def blp(value):
    x = value[0:value.index('/')]
    y = value[value.index('/')+1:len(value)]
    if int(x)>140 and int(y)>90:
        return 'High'
    elif int(x)<100 and int(y)<70:
        return 'Low'
    else :
        return 'Normal'

df['blood_pressure_cat'] = df['Blood Pressure'].apply(blp)
#df.head()

# Replacing target variable for better visualizations"""

df['Heart Attack Risk'] = df['Heart Attack Risk'].replace(to_replace = [0,1],value=['no','yes'])

#### Heart Attack Risk column is considered as Integer which may not give better visualizations, For getting the better visualizations we need to convert it into an object. Hence 0,1 is replaced with No and Yes respectively"""

#df.info()

# Visualizations

#### Selecting the Numerical columns first

df_num = df[['Age','Cholesterol','Heart Rate','Exercise Hours Per Week','Sedentary Hours Per Day','Income','BMI','Triglycerides']]
#df_num.head()

### Creating a Grid of subplots to display Boxplots to visualize the relationship between Numerical features and the Heart Attack Risk in a dataset"""
'''
fig,ax=plt.subplots(nrows=4,ncols=2,figsize=(20,30))
for v,s in zip(df_num.columns,ax.flatten()):
    sns.boxplot(x=df['Heart Attack Risk'],y=df_num[v],ax=s)
plt.show()
'''
### <i> From the above box plot visuals, It is evident that there is no significant correlation between numerical features in the dataset and to the Heart Attack Risk."""

#df.columns
### Separating the categorical or non-numeric data from the DataFrame"""

df_cat = df.drop(['Age','Cholesterol','Heart Rate','Exercise Hours Per Week','Sedentary Hours Per Day',
                  'Income','BMI','Triglycerides','Patient ID','Blood Pressure','Heart Attack Risk'],axis=1)
#df_cat.head()

### Creating a Grid of subplots to display Histogram plots to visualize the relationship between Categorical features and the Heart Attack Risk in the dataset."""
'''
fig,ax=plt.subplots(nrows=8,ncols=2,figsize=(20,30))
for v,s in zip(df_cat.columns,ax.flatten()):
    sns.histplot(x=df['Heart Attack Risk'],hue=df_cat[v],stat='percent',multiple='dodge',ax=s)
plt.show()
'''
### <i>The imbalance in the dataset's classes within the 'Heart Attack Risk' and other categorical variables hinders our ability to predict results effectively. This imbalance can lead to Bias in Predictions,Misleading Accuracy, Inadequate Learning."""

df_num_col = df.select_dtypes(include=['int64', 'float64'])
#df_num_col

#df.info()

## Encoding the Categorical Variables

#### Machine learning models require numerical input. Hence, categorical variables need to be transformed or encoded into numerical representations. Label Encoding assigns a unique integer to each category.


from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])
df['Diet'] = label_encoder.fit_transform(df['Diet'])
df['Country'] = label_encoder.fit_transform(df['Country'])
df['Continent'] = label_encoder.fit_transform(df['Continent'])
df['Hemisphere'] = label_encoder.fit_transform(df['Hemisphere'])
df['blood_pressure_cat'] = label_encoder.fit_transform(df['blood_pressure_cat'])
df['Heart Attack Risk'] = label_encoder.fit_transform(df['Heart Attack Risk'])

## Dependent and Independent Variables

#### The dependent variable is the outcome or response that is being studied and measured whereas Independent variable is the variable that is used to predict or explain the changes in the dependent variable.

y = df['Heart Attack Risk']
X = df[['Age', 'Sex', 'Cholesterol',
       'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
       'Previous Heart Problems', 'Medication Use', 'Stress Level',
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Country',
       'Continent', 'Hemisphere','blood_pressure_cat']]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

## Predicting the Significant and Insignificant Variables using Logistic Regression
#### Significant variable has a notable impact on the probability of the occurrence of the outcome variable whereas Changes in the value of the insignificant variable are not associated with changes in the likelihood of the outcome variable.


lr_model=sm.Logit(y_train,x_train).fit()
#lr_model.summary()

#### In general, if p value is less than 5%, those variables are considered as significant and the variables are having p value less than 5% and

## Spliting the Data by considering the significant variables

## Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
classifier_logistic = LogisticRegression()
lr_model = classifier_logistic.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
lr_acc = round(accuracy_score(y_test, y_pred), 3)
lr_matrix = confusion_matrix(y_test, y_pred)
yy_prob = lr_model.predict_proba(x_test)

lr_prec = precision_score(y_test,y_pred)
lr_rec = recall_score(y_test,y_pred)
lr_train_accuracy=accuracy_score(y_train,lr_model.predict(x_train))
lr_test_accuracy=accuracy_score(y_test,y_pred)

'''
print(f'Train Data Accuracy of Logistic Regression model is {lr_train_accuracy * 100}%')
print(f'Test Data Accuracy of Logistic Regression model is {lr_test_accuracy * 100}%')
print(f'Precision of Logistic Regression model is {lr_prec * 100}%')
print(f'Recall of Logistic Regression model is {lr_rec * 100}%')
print("Confusion Matrix:\n", lr_matrix)
print("Predicted Probabilities:\n", yy_prob)
'''

## KNN Classification"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

yy = df['Heart Attack Risk']
xx = df[['Heart Rate', 'Diabetes', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Previous Heart Problems', 'Stress Level', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Hemisphere']]
xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, random_state=10, test_size=0.3)
knn = KNeighborsClassifier()
knn_model=knn.fit(xx_train, yy_train)
knn_y_pred = knn_model.predict(xx_test)
knn_conf_matrix = confusion_matrix(yy_test,knn_y_pred)

knn_prec = precision_score(yy_test, knn_y_pred)
knn_rec = recall_score(yy_test,knn_y_pred)
knn_train_accuracy=accuracy_score(yy_train,knn_model.predict(xx_train))
knn_test_accuracy=accuracy_score(yy_test,knn_y_pred)

'''
print(f'Train Data Accuracy of KNN Classifier model is {knn_train_accuracy * 100}%')
print(f'Test Data Accuracy of KNN Classifier model is {knn_test_accuracy * 100}%')
print(f'Precision of KNN forest classifier model is {knn_prec * 100}%')
print(f'Recall of KNN forest classifier model is {knn_rec * 100}%')
'''

## XG Boost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm

xgb = XGBClassifier()
xgb_model=xgb.fit(x_train, y_train)
pred_test_xgboost = xgb_model.predict(x_test)
xgboost_matrix = confusion_matrix(y_test, pred_test_xgboost)
'''
print("Confusion Matrix for XGBoost:")
print(xgboost_matrix)
'''
xg_prec = precision_score(y_test, pred_test_xgboost)
xg_rec = recall_score(y_test,pred_test_xgboost)
xg_train_accuracy=accuracy_score(y_train,xgb_model.predict(x_train))
xg_test_accuracy=accuracy_score(y_test,pred_test_xgboost)

'''
print(f'Train Data Accuracy of XGBoost classifier model is {xg_train_accuracy * 100}%')
print(f'Test Data Accuracy of XGBoost classifier model is {xg_test_accuracy * 100}%')
print(f'Precision of XGBoost model is {xg_prec * 100}%')
print(f'Recall of XGBoost model is {xg_rec * 100}%')
'''

## Decicsion Tree"""

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt_model = dt.fit(x_train, y_train)
dt_y_pred = dt_model.predict(x_test)
tree_matrix = confusion_matrix(y_test, dt_y_pred)

dt_prec = precision_score(y_test, dt_y_pred)
dt_rec = recall_score(y_test,dt_y_pred)
dt_train_accuracy=accuracy_score(y_train,dt_model.predict(x_train))
dt_test_accuracy=accuracy_score(y_test,dt_y_pred)

'''
print(f'Test Data Accuracy of Decision Tree classifier model is {dt_test_accuracy * 100}%')
print(f'Precision of Decision tree model is {dt_prec * 100}%')
print(f'Recall of Decision tree model is {dt_rec * 100}%')
'''

## Plotting the Tree"""
'''
from sklearn.tree import plot_tree
feature_names_list = df.columns.tolist()
plt.figure(figsize=(20,10))
plot_tree(dt_model, filled= True)
plt.show()
'''

## RandomForest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_prec = precision_score(y_test, rf_y_pred)
rf_rec = recall_score(y_test,rf_y_pred)
rf_train_accuracy=accuracy_score(y_train,rf_model.predict(x_train))
rf_test_accuracy=accuracy_score(y_test,rf_y_pred)

'''
print(f'Train Data Accuracy of XGBoost classifier model is {rf_train_accuracy * 100}%')
print(f'Test Data Accuracy of XGBoost classifier model is {rf_test_accuracy * 100}%')
print(f'Precision of Random forest classifier model is {rf_prec * 100}%')
print(f'Recall of Random forest classifier model is {rf_rec * 100}%')
'''
## Adaboost Classifier"""

from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier()
ada_model=ada.fit(x_train,y_train)
y_pred_ada=ada_model.predict(x_test)
accuracy_ada=accuracy_score(y_test,y_pred_ada)
print("The accuracy score is :",accuracy_ada)
ada_train_accuracy=accuracy_score(y_train,ada_model.predict(x_train))
ada_test_accuracy=accuracy_score(y_test,y_pred_ada)
ada_matrix = confusion_matrix(y_test, y_pred_ada)

'''
print(f'Train Data Accuracy of XGBoost classifier model is {ada_train_accuracy * 100}%')
print(f'Test Data Accuracy of XGBoost classifier model is {ada_test_accuracy * 100}%')
'''

prec_ada=precision_score(y_test,y_pred_ada)
recall_ada = recall_score(y_test,y_pred_ada)
'''
print("The precision of the model is :",prec_ada)
print("The Recall Score of the model is :",recall_ada)
'''
## GradientBoostingClassifier"""

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc_model=gbc.fit(x_train,y_train)
y_pred_gbc=gbc_model.predict(x_test)
train_accuracy_gbc=accuracy_score(y_train,gbc_model.predict(x_train))
test_accuracy_gbc=accuracy_score(y_test,y_pred_gbc)
#gbc_matrix = confusion_matrix(y_test, y_pred_gbc)

'''
print("The accuracy score for the train data is :",train_accuracy_gbc)
print("The accuracy score of test data is :",test_accuracy_gbc)
prec_gbc=precision_score(y_test,y_pred_gbc)
recall_gbc=recall_score(y_test,y_pred_gbc)
print("The precision of the model is :",prec_gbc)
print("The Recall score of the model is :",recall_gbc)
'''
### Creating a Dataframe which portrays the Accuracy, Precision and Recall Scores of all Models"""
'''
m=["Logistic regression","K-neighbors", "Decision Tree", "Random Forest",'XGBoost','Adaboost','Gradient Boosting']
tea= [lr_test_accuracy,knn_test_accuracy,dt_test_accuracy,rf_test_accuracy,xg_test_accuracy,ada_test_accuracy,test_accuracy_gbc]
p=[lr_prec,knn_prec,dt_prec,rf_prec,xg_prec,prec_ada,prec_gbc]
r=[lr_rec,knn_rec,dt_rec,rf_rec,xg_rec,recall_ada,recall_gbc]

model_df2=pd.DataFrame({"model":m,"test accuracy": tea,"precision":p,"recall Score":r})
model_df2

colors = ['red', 'orange', 'green', 'red', 'blue', 'purple', 'brown']
plt.figure(figsize=(8, 6))
bars = plt.bar(m, r, color = colors)
plt.xlabel('Classifiers')
plt.ylabel('Recall Scores')
plt.title('Recall Scores for Different Classifiers')
plt.bar_label(bars, labels=[f"{score:.2f}" for score in r], label_type='edge', color='black')
plt.ylim(0, 0.5)  # Set the y-axis limit from 0 to 1 (as recall scores are between 0 and 1)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

## <i> <font color='black'> <div style='background-color:yellow'> Among the above seven models considered, Decision Tree Model was identified as the most suitable choice due to its balanced performance across various metrics like Accuracy, Precision and Recall scores.

## <i> <font color='black'> <div style='background-color:yellow'> Decision Tree exhibited a favorable recall score, emphasizing its proficiency in capturing relevant instances within the dataset, thus minimizing false negatives.

## Decision Tree Feature importance scores


df_dt_sig=pd.DataFrame({"features":X.columns,"significance":dt_model.feature_importances_})
df_dt_sig=df_dt_sig.sort_values(by='significance',ascending=False)
df_dt_sig

### <i><font color='black'> <div style='background-color:yellow'> The feature importances attribute represents the relative importance of each feature in the model. The values provides a quick way to identify the most influential features used by the model during the training process.  Sedentary Hours Per Day, BMI, Triglycerides features have more impact on the model's predictions. More over Sedentary Hours Per Day, BMI, Triglycerides are more important in the model's decision-making process.

### Feature Importance Scores of all classifiers

df_rf_sig=pd.DataFrame({"features":X.columns,"significance":rf_model.feature_importances_})
df_rf_sig=df_rf_sig.sort_values(by='significance',ascending=False)
df_gb_sig=pd.DataFrame({"features":X.columns,"significance":gbc_model.feature_importances_})
df_gb_sig=df_gb_sig.sort_values(by='significance',ascending=False)
df_xg_sig=pd.DataFrame({"features":X.columns,"significance":xgb_model.feature_importances_})
df_xg_sig=df_xg_sig.sort_values(by='significance',ascending=False)
df_ada_sig=pd.DataFrame({"features":X.columns,"significance":ada_model.feature_importances_})
df_ada_sig=df_ada_sig.sort_values(by='significance',ascending=False)
df2= pd.concat([df_dt_sig,df_gb_sig,df_rf_sig,df_ada_sig,df_xg_sig],axis=1)
df2

### <i><font color='black'> <div style='background-color:yellow'> From all the above Importance scores of all the Models considered for Analysis, Sedentary Hours Per Day, BMI, Triglycerides have more impact on the model's predictions.

## Hyperparamter Tuning using GridSearchCV Method
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_dt_model = grid_search.best_estimator_
dt_y_pred = best_dt_model.predict(x_test)
tree_matrix = confusion_matrix(y_test, dt_y_pred)
dtg5a_acc = round(accuracy_score(y_test, dt_y_pred), 3)
dtg5a_prec = precision_score(y_test, dt_y_pred)
dtg5a_rec = recall_score(y_test, dt_y_pred)

'''
print(f'Accuracy of Decision tree model is {dtg5a_acc * 100}%')
print(f'Precision of Decision tree model is {dtg5a_prec * 100}%')
print(f'Recall of Decision tree model is {dtg5a_rec * 100}%')
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='recall')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_dt_model = grid_search.best_estimator_
dt_y_pred = best_dt_model.predict(x_test)
treegs_matrix = confusion_matrix(y_test, dt_y_pred)
dtgs5r_acc = round(accuracy_score(y_test, dt_y_pred), 3)
dtgs5r_prec = precision_score(y_test, dt_y_pred)
dtgs5r_rec = recall_score(y_test, dt_y_pred)
'''
print(f'Accuracy of Decision tree model is {dtgs5r_acc * 100}%')
print(f'Precision of Decision tree model is {dtgs5r_prec * 100}%')
print(f'Recall of Decision tree model is {dtgs5r_rec * 100}%')

## Creating a Dataframe to compare the Metrics with Hyperparameter Tuning Method"""

m1=["Decision Tree", "GridSearchCV"]
a1=[dt_test_accuracy,dtgs5r_acc]
p1=[dt_prec,dtgs5r_prec]
r1=[dt_rec,dtgs5r_rec]

model_df3=pd.DataFrame({"model":m1,"accuracy":a1,"precision":p1,"recall Score":r1})
model_df3

## Confusion Matrix of Decision Tree, KNN, Logistic Regression, Random Forest, Adaboost, Gradient Boosting Models"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import seaborn as sns

# %matplotlib inline
rcParams['figure.figsize'] = 10,8
fig, ax = plt.subplots(4,2)
sns.heatmap(tree_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[0,0])
ax[0,0].set_xlabel('Predicted')
ax[0,0].set_ylabel('Actual')
ax[0,0].set_title('Decision Tree Confusion Matrix')

sns.heatmap(rf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[0,1])
ax[0,1].set_xlabel('Predicted')
ax[0,1].set_ylabel('Actual')
ax[0,1].set_title('Random Forest Confusion Matrix')

sns.heatmap(xgboost_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[1,0])
ax[1,0].set_xlabel('Predicted')
ax[1,0].set_ylabel('Actual')
ax[1,0].set_title('XGBoost Confusion Matrix')

sns.heatmap(lr_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[1,1])
ax[1,1].set_xlabel('Predicted')
ax[1,1].set_ylabel('Actual')
ax[1,1].set_title('Logistic Regression Confusion Matrix')

sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[2,0])
ax[2,0].set_xlabel('Predicted')
ax[2,0].set_ylabel('Actual')
ax[2,0].set_title('KNN Classification Confusion Matrix')

sns.heatmap(ada_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[2,1])
ax[2,1].set_xlabel('Predicted')
ax[2,1].set_ylabel('Actual')
ax[2,1].set_title('Adaboost Classification Confusion Matrix')

sns.heatmap(gbc_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], ax=ax[3,0])
ax[3,0].set_xlabel('Predicted')
ax[3,0].set_ylabel('Actual')
ax[3,0].set_title('Gradient Boosting Classification Confusion Matrix')


blank_image = np.ones((100, 100, 3))
blank_image.fill(1)
ax[3, 1].imshow(blank_image)
ax[3, 1].axis('off')

plt.tight_layout()
plt.show()

# <u>Conclusion
### 1.Among the  seven models considered, Decision Tree Model is identified as the most suitable choice due to its balanced performance across various metrics like Accuracy, Precision and Recall scores.
### 2.Sedentary Hours Per Day, BMI, Triglycerides have more impact on the model's predictions.
### 3.From the bar graph, we can compare the Recall scores and can conclude the best model which is the Decision Tree model.
'''

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 120, 30)
    cholesterol = st.sidebar.slider('Cholesterol', 0, 500, 100)
    heart_rate = st.sidebar.slider('Heart Rate', 0, 150, 72)
    exercise_hours = st.sidebar.slider('Exercise Hours per week', 0, 30, 12 )
    sedentary_hours = st.sidebar.slider('Sedentary Hours per Day', 0, 20, 10)
    income = st.sidebar.slider('Income', 0, 400000, 50000)
    bmi = st.sidebar.slider('BMI', 0, 50, 20)
    triglycerides = st.sidebar.slider('Triglycerides', 0, 1000, 500)
    sex = st.sidebar.selectbox('Sex (0 = Male, 1 = Female)', [0, 1])
    Diabetes = st.sidebar.selectbox('Diabetes (0 = No, 1 = Yes)', [0, 1])
    family_history = st.sidebar.selectbox('family_history (0 = No, 1 = Yes)', [0, 1])
    obesity = st.sidebar.selectbox('Obesity (0 = No, 1 = Yes)', [0, 1])
    alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption (0 = No, 1 = Yes)', [0, 1])
    hemisphere = st.sidebar.selectbox('Smoker (0 = No, 1 = Yes)', [0, 1])
    country = st.sidebar.selectbox('Select a Country (0–19)', options=list(range(20))) 
    continent = st.sidebar.selectbox('Select a Continent (0–5)', options=list(range(6))) 
    Diet = st.sidebar.selectbox('Diet (0–2)', options=list(range(3)))
    data = {'age': age,
            'cholesterol': cholesterol,
            'heart_rate': heart_rate,
            'exercise_hours': exercise_hours_per_week,
            'sedentary_hours': sedentary_hours,
            'income': income,
            'bmi': bmi,
            'sex': sex,
            'Diabetes': Diabetes,
            'family_history': family_history,
            'alcohol_consumption': alcohol_consumption,
            'hemisphere': hemisphere,
            'country': country,
            'continent': continent,
            'Diet': Diet,
            'obesity': obesity,
            'triglycerides': triglycerides}
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input_features()

st.subheader('User Input parameters')
st.write(df1)

prediction = rf_model.predict(df1)
prediction_proba = rf_model.predict_proba(df1)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


