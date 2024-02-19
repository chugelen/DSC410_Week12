#import libraries
import dash
from dash import dcc,html
from dash.dependencies import Input, Output

import tensorflow as tf
from keras.models import load_model
import joblib

import numpy as np
import pandas as pd

app = dash.Dash(__name__)
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

model = load_model('./Model/attrition_prediction_model.h5')
scaler = joblib.load('./Model/scaler.joblib')
#model._make_predict_function()  # I kept getting a error with this code, according to updates this is no longer needed

## Div for age
input_age = dcc.Input(
    id='Age',        
    placeholder='',
    type='number',
    value='0')

div_Age = html.Div(
        children=[html.H3('Age:'), input_age],
        className="four columns"
        )

## Div for DailyRate
input_DailyRate = dcc.Input(
    id='DailyRate',
    placeholder = '',
    type='number',
    value='0')

div_DailyRate = html.Div(
        children=[html.H3('DailyRate:'), input_DailyRate],
        className="four columns"
        )

# Div for DistanceFromHome
input_DistanceFromHome = dcc.Input(
    id='DistanceFromHome',
    placeholder='',
    type='number',
    value='0'
)

div_DistanceFromHome = html.Div(
    children=[html.H3('DistanceFromHome:'), input_DistanceFromHome],
    className="four columns"
)

# Div for Education
input_Education = dcc.Input(
    id='Education',
    placeholder='',
    type='number',
    value='0'
)

div_Education = html.Div(
    children=[html.H3('Education:'), input_Education],
    className="four columns"
)

# Div for EnvironmentSatisfaction
input_EnvironmentSatisfaction = dcc.Input(
    id='EnvironmentSatisfaction',
    placeholder='',
    type='number',
    value='0'
)

div_EnvironmentSatisfaction = html.Div(
    children=[html.H3('EnvironmentSatisfaction:'), input_EnvironmentSatisfaction],
    className="four columns"
)

# Div for HourlyRate
input_HourlyRate = dcc.Input(
    id='HourlyRate',
    placeholder='',
    type='number',
    value='0'
)

div_HourlyRate = html.Div(
    children=[html.H3('HourlyRate:'), input_HourlyRate],
    className="four columns"
)

# Div for JobInvolvement
input_JobInvolvement = dcc.Input(
    id='JobInvolvement',
    placeholder='',
    type='number',
    value='0'
)

div_JobInvolvement = html.Div(
    children=[html.H3('JobInvolvement:'), input_JobInvolvement],
    className="four columns"
)

# Div for JobSatisfaction
input_JobSatisfaction = dcc.Input(
    id='JobSatisfaction',
    placeholder='',
    type='number',
    value='0'
)

div_JobSatisfaction = html.Div(
    children=[html.H3('JobSatisfaction:'), input_JobSatisfaction],
    className="four columns"
)

# Div for MonthlyIncome
input_MonthlyIncome = dcc.Input(
    id='MonthlyIncome',
    placeholder='',
    type='number',
    value='0'
)

div_MonthlyIncome = html.Div(
    children=[html.H3('MonthlyIncome:'), input_MonthlyIncome],
    className="four columns"
)

# Div for MonthlyRate
input_MonthlyRate = dcc.Input(
    id='MonthlyRate',
    placeholder='',
    type='number',
    value='0'
)

div_MonthlyRate = html.Div(
    children=[html.H3('MonthlyRate:'), input_MonthlyRate],
    className="four columns"
)

# Div for NumCompaniesWorked
input_NumCompaniesWorked = dcc.Input(
    id='NumCompaniesWorked',
    placeholder='',
    type='number',
    value='0'
)

div_NumCompaniesWorked = html.Div(
    children=[html.H3('NumCompaniesWorked:'), input_NumCompaniesWorked],
    className="four columns"
)

# Div for PerformanceRating
input_PerformanceRating = dcc.Input(
    id='PerformanceRating',
    placeholder='',
    type='number',
    value='0'
)

div_PerformanceRating = html.Div(
    children=[html.H3('PerformanceRating:'), input_PerformanceRating],
    className="four columns"
)

# Div for RelationshipSatisfaction
input_RelationshipSatisfaction = dcc.Input(
    id='RelationshipSatisfaction',
    placeholder='',
    type='number',
    value='0'
)

div_RelationshipSatisfaction = html.Div(
    children=[html.H3('RelationshipSatisfaction:'), input_RelationshipSatisfaction],
    className="four columns"
)

# Div for StockOptionLevel
input_StockOptionLevel = dcc.Input(
    id='StockOptionLevel',
    placeholder='',
    type='number',
    value='0'
)

div_StockOptionLevel = html.Div(
    children=[html.H3('StockOptionLevel:'), input_StockOptionLevel],
    className="four columns"
)

# Div for TrainingTimesLastYear
input_TrainingTimesLastYear = dcc.Input(
    id='TrainingTimesLastYear',
    placeholder='',
    type='number',
    value='0'
)

div_TrainingTimesLastYear = html.Div(
    children=[html.H3('TrainingTimesLastYear:'), input_TrainingTimesLastYear],
    className="four columns"
)

# Div for WorkLifeBalance
input_WorkLifeBalance = dcc.Input(
    id='WorkLifeBalance',
    placeholder='',
    type='number',
    value='0'
)

div_WorkLifeBalance = html.Div(
    children=[html.H3('WorkLifeBalance:'), input_WorkLifeBalance],
    className="four columns"
)

# Div for YearsAtCompany
input_YearsAtCompany = dcc.Input(
    id='YearsAtCompany',
    placeholder='',
    type='number',
    value='0'
)

div_YearsAtCompany = html.Div(
    children=[html.H3('YearsAtCompany:'), input_YearsAtCompany],
    className="four columns"
)

## Div for BusinessTravel
BusinessTravel_values = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
BusinessTravel_options = [{'label': x, 'value': x} for x in BusinessTravel_values]
input_BusinessTravel = dcc.Dropdown(
    id='BusinessTravel',        
    options = BusinessTravel_options,
    value = 'Non-Travel'             
    )

div_BusinessTravel = html.Div(
        children=[html.H3('BusinessTravel:'), input_BusinessTravel],
        className="four columns"
        )

# BusinessTravel encoding
Business_Travel_encoding = {
    'Non-Travel': 0,
    'Travel_Frequently': 1,
    'Travel_Rarely': 2
}

## Div for Department
Department_values = ['Sales', 'Research & Development', 'Human Resources']
Department_options = [{'label': x, 'value': x} for x in Department_values]
input_Department = dcc.Dropdown(
    id='Department',        
    options = Department_options,
    value = 'Research & Development'             
    )

div_Department = html.Div(
        children=[html.H3('Department:'), input_Department],
        className="four columns"
        )

# Department encoding
Department_encoding = {
    'Human Resources': 0,
    'Research & Development': 1,
    'Sales': 2
}

## Div for EducationField
EducationField_values = ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree',
 'Human Resources']
EducationField_options = [{'label': x, 'value': x} for x in EducationField_values]
input_EducationField = dcc.Dropdown(
    id='EducationField',        
    options = EducationField_options,
    value = 'Life Sciences'             
    )

div_EducationField = html.Div(
        children=[html.H3('EducationField:'), input_EducationField],
        className="four columns"
        )

# EducationField encoding
EducationField_encoding = {    
    'Human Resources':0,
    'Life Sciences': 1,
    'Marketing': 2,
    'Medical': 3,
    'Other': 4,
    'Technical Degree': 5
}

## Div for Gender
Gender_values = ['Female', 'Male']
Gender_options = [{'label': x, 'value': x} for x in Gender_values]
input_Gender = dcc.Dropdown(
    id='Gender',        
    options = Gender_options,
    value ='Male'             
    )

div_Gender = html.Div(
        children=[html.H3('Gender:'), input_Gender],
        className="four columns"
        )

# Gender encoding
Gender_encoding = {    
    'Female':0,
    'Male': 1
}

## Div for JobRole
JobRole_values = ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                  'Manufacturing Director', 'Healthcare Representative', 'Manager',
                  'Sales Representative', 'Research Director', 'Human Resources']
JobRole_options = [{'label': x, 'value': x} for x in JobRole_values]
input_JobRole = dcc.Dropdown(
    id='JobRole',        
    options = JobRole_options,
    value = 'Sales Executive'             
    )

div_JobRole = html.Div(
        children=[html.H3('JobRole:'), input_JobRole],
        className="four columns"
        )

# JobRole encoding
JobRole_encoding = {    
    'Sales Executive': 7,
    'Research Scientist': 6,
    'Laboratory Technician': 2,
    'Manufacturing Director': 4,
    'Healthcare Representative': 0,
    'Manager': 3,
    'Sales Representative': 8,
    'Research Director': 5,
    'Human Resources': 1
}

## Div for MaritalStatus
MaritalStatus_values =  ['Single', 'Married', 'Divorced']
MaritalStatus_options = [{'label': x, 'value': x} for x in MaritalStatus_values]
input_MaritalStatus = dcc.Dropdown(
    id='MaritalStatus',        
    options = MaritalStatus_options,
    value = 'Married'             
    )

div_MaritalStatus = html.Div(
        children=[html.H3('MaritalStatus:'), input_MaritalStatus],
        className="four columns"
        )

# MaritalStatus encoding
MaritalStatus_encoding = {    
    'Divorced':0,
    'Married': 1,
    'Single': 2
}

## Div for OverTime
OverTime_values =  ['No','Yes']
OverTime_options = [{'label': x, 'value': x} for x in OverTime_values]
input_OverTime = dcc.Dropdown(
    id='OverTime',        
    options = OverTime_options,
    value = 'No'             
    )

div_OverTime = html.Div(
        children=[html.H3('OverTime:'), input_OverTime],
        className="four columns"
        )

# OverTime encoding
OverTime_encoding = {    
    'No':0,
    'Yes': 1
}

## Div for numerical characteristics
div_numerical = html.Div(
        children = [div_Age, div_DailyRate, div_DistanceFromHome, div_Education, div_EnvironmentSatisfaction, 
                    div_HourlyRate, div_JobInvolvement, div_JobSatisfaction, div_MonthlyIncome, div_MonthlyRate, 
                    div_NumCompaniesWorked, div_PerformanceRating, div_RelationshipSatisfaction, div_StockOptionLevel, 
                    div_TrainingTimesLastYear, div_WorkLifeBalance, div_YearsAtCompany],
        className="row"
        )

## Div for categorical
div_categorical = html.Div(
        children = [div_BusinessTravel, div_Department, div_EducationField, div_Gender, div_JobRole, div_MaritalStatus, div_OverTime],
        className="row"
       )

def get_prediction(Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, 
                    EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobRole, JobSatisfaction, MaritalStatus, 
                    MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PerformanceRating, RelationshipSatisfaction, 
                    StockOptionLevel, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany):
        
    cols = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'Attrition']
        
    ## produce a dataframe with a single row of zeros
    df = pd.DataFrame(data = np.zeros((1,len(cols))), columns = cols)    
    
    # get the numeric characteristics    
    df.loc[0,'Age'] = Age
    df.loc[0,'DailyRate'] = DailyRate
    df.loc[0, 'DistanceFromHome'] = DistanceFromHome
    df.loc[0, 'Education'] = Education
    df.loc[0, 'EnvironmentSatisfaction'] = EnvironmentSatisfaction
    df.loc[0, 'HourlyRate'] = HourlyRate
    df.loc[0, 'JobInvolvement'] = JobInvolvement
    df.loc[0, 'JobSatisfaction'] = JobSatisfaction
    df.loc[0, 'MonthlyIncome'] = MonthlyIncome
    df.loc[0, 'MonthlyRate'] = MonthlyRate
    df.loc[0, 'NumCompaniesWorked'] = NumCompaniesWorked
    df.loc[0, 'PerformanceRating'] = PerformanceRating
    df.loc[0, 'RelationshipSatisfaction'] = RelationshipSatisfaction
    df.loc[0, 'StockOptionLevel'] = StockOptionLevel
    df.loc[0, 'TrainingTimesLastYear'] = TrainingTimesLastYear
    df.loc[0, 'WorkLifeBalance'] = WorkLifeBalance
    df.loc[0, 'YearsAtCompany'] = YearsAtCompany
    
    # encoding for categorial features
    df['BusinessTravel'] = Business_Travel_encoding[BusinessTravel]    
    df['Department'] = Department_encoding[Department]
    df['EducationField'] = EducationField_encoding[EducationField]
    df['Gender'] = Gender_encoding[Gender]
    df['JobRole'] = JobRole_encoding[JobRole]
    df['MaritalStatus'] = MaritalStatus_encoding[MaritalStatus]
    df['OverTime'] = OverTime_encoding[OverTime]
   
    # Set 'Claim' to default value of 0 since it will be predicted
    df['Attrition'] = 0       
     
    # Scale the features using the trained scaler
    df_scaled = scaler.transform(df.drop('Attrition', axis=1))
    
    ## Get the predictions using our trained neural network
    prediction = model.predict(df_scaled).flatten()[0]
        
    ## Prediction to binary (claim or no claim) based on a threshold
    threshold = 0.558 
    threshold_value = prediction
    prediction = 1 if prediction >= threshold else 0   
        
    # Set 'Attrition' to predicted value
    df['Attrition'] = prediction
    print('Threshold Value:',threshold_value)
    return prediction
    
#App layout
app.layout = html.Div([
    html.H1('Attrition Prediction'),

    html.H2('Enter the employee details to predict the attrition probability'),

    html.Div(
        children=[div_numerical, div_categorical]
    ),

    html.H1(id='output', style={'margin-top': '50px', 'text-align': 'center'})
])

# Predictor features for the callback
predictors = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany']

# Callback function to update the output
@app.callback(
    Output('output', 'children'),
    [Input(x, 'value') for x in predictors])

def show_prediction(Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, 
                    EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobRole, JobSatisfaction, MaritalStatus, 
                    MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PerformanceRating, RelationshipSatisfaction, 
                    StockOptionLevel, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany):
    pred = get_prediction(Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, 
                    EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobRole, JobSatisfaction, MaritalStatus, 
                    MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PerformanceRating, RelationshipSatisfaction, 
                    StockOptionLevel, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany)
    
    prediction_value = 'Yes' if pred == 1 else 'No'
    
    return f"Attrition Prediction: {'Yes' if pred == 1 else 'No'}" 
    
if __name__ == '__main__':
    app.run_server(debug=True)