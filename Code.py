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

### <i> From the above box plot visuals, It is evident that there is no significant correlation between numerical features in the dataset and to the Heart Attack Risk."""

#df.columns
### Separating the categorical or non-numeric data from the DataFrame"""

df_cat = df.drop(['Age','Cholesterol','Heart Rate','Exercise Hours Per Week','Sedentary Hours Per Day',
                  'Income','BMI','Triglycerides','Patient ID','Blood Pressure','Heart Attack Risk'],axis=1)
#df_cat.head()

### Creating a Grid of subplots to display Histogram plots to visualize the relationship between Categorical features and the Heart Attack Risk in the dataset."""
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


## XG Boost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm

xgb = XGBClassifier()
xgb_model=xgb.fit(x_train, y_train)
pred_test_xgboost = xgb_model.predict(x_test)
xgboost_matrix = confusion_matrix(y_test, pred_test_xgboost)
xg_prec = precision_score(y_test, pred_test_xgboost)
xg_rec = recall_score(y_test,pred_test_xgboost)
xg_train_accuracy=accuracy_score(y_train,xgb_model.predict(x_train))
xg_test_accuracy=accuracy_score(y_test,pred_test_xgboost)


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

## Plotting the Tree"""

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


prec_ada=precision_score(y_test,y_pred_ada)
recall_ada = recall_score(y_test,y_pred_ada)
## GradientBoostingClassifier"""

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc_model=gbc.fit(x_train,y_train)
y_pred_gbc=gbc_model.predict(x_test)
train_accuracy_gbc=accuracy_score(y_train,gbc_model.predict(x_train))
test_accuracy_gbc=accuracy_score(y_test,y_pred_gbc)
#gbc_matrix = confusion_matrix(y_test, y_pred_gbc)

### Creating a Dataframe which portrays the Accuracy, Precision and Recall Scores of all Models"""

## <i> <font color='black'> <div style='background-color:yellow'> Among the above seven models considered, Decision Tree Model was identified as the most suitable choice due to its balanced performance across various metrics like Accuracy, Precision and Recall scores.

## <i> <font color='black'> <div style='background-color:yellow'> Decision Tree exhibited a favorable recall score, emphasizing its proficiency in capturing relevant instances within the dataset, thus minimizing false negatives.

## Decision Tree Feature importance scores



### <i><font color='black'> <div style='background-color:yellow'> From all the above Importance scores of all the Models considered for Analysis, Sedentary Hours Per Day, BMI, Triglycerides have more impact on the model's predictions.

## Hyperparamter Tuning using GridSearchCV Method

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



prediction = rf_model.predict(df1)
prediction_proba = rf_model.predict_proba(df1)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


