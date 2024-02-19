#import libraries
import dash
from dash import dcc,html
from dash.dependencies import Input, Output

import tensorflow as tf
from keras.models import load_model
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

app = dash.Dash(__name__)
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

model = load_model('./Model/travel_insurance_claim_model.h5')
scaler = joblib.load('./Model/scaler.joblib')
#model._make_predict_function()  # I kept getting a error with this code, according to updates this is no longer needed


## Div for duration
input_duration = dcc.Input(
    id='duration',
    placeholder = '',
    type='number',
    value='0')

div_duration = html.Div(
        children=[html.H3('Duration:'), input_duration],
        className="four columns"
        )

## Div for Net sales
input_sales = dcc.Input(
    id='sales',
    placeholder='',
    type='number',
    value='0')

div_sales = html.Div(
        children=[html.H3('Net Sale:'), input_sales],
        className="four columns"
        )

## Div for age
input_age = dcc.Input(
    id='age',        
    placeholder='',
    type='number',
    value='0')

div_age = html.Div(
        children=[html.H3('Age:'), input_age],
        className="four columns"
        )

## Div for agency
agency_values = ['CBH', 'CWT', 'JZI', 'KML', 'EPX', 'C2B', 'JWT', 'RAB', 'SSI', 'ART', 'CSR','CCR',
 'ADM', 'LWC', 'TTW', 'TST']
agency_options = [{'label': x, 'value': x} for x in agency_values]
input_agency = dcc.Dropdown(
    id='agency',
    options = agency_options,
    value = 'CBH'
    )

div_agency = html.Div(
        children=[html.H3('Agency:'), input_agency],
        className="four columns"
        )

## Div for product
product_values = ['Comprehensive Plan', 'Rental Vehicle Excess Insurance', 'Value Plan', 
                  'Basic Plan', 'Premier Plan','2 way Comprehensive Plan', 'Bronze Plan',
                  'Silver Plan', 'Annual Silver Plan', 'Cancellation Plan','1 way Comprehensive Plan',
                  'Ticket Protector', '24 Protect', 'Gold Plan', 'Annual Gold Plan',
                  'Single Trip Travel Protect Silver', 'Individual Comprehensive Plan', 
                  'Spouse or Parents Comprehensive Plan', 'Annual Travel Protect Silver', 
                  'Single Trip Travel Protect Platinum', 'Annual Travel Protect Gold', 
                  'Single Trip Travel Protect Gold', 'Annual Travel Protect Platinum', 
                  'Child Comprehensive Plan', 'Travel Cruise Protect', 'Travel Cruise Protect Family']
product_options = [{'label': x, 'value': x} for x in product_values]
input_product = dcc.Dropdown(
    id='product',        
    options = product_options,
    value = 'Basic Plan'             
    )

div_product = html.Div(
        children=[html.H3('Product:'), input_product],
        className="four columns"
        )

# product encoding
plan_mapping = {
    # Group: Comprehensive Plans
    'Comprehensive Plans': [
        '1 way Comprehensive Plan',
        '2 way Comprehensive Plan',
        'Comprehensive Plan',                        
        'Spouse or Parents Comprehensive Plan',
        'Child Comprehensive Plan',
    ],

    # Group: Leveled Plans
    'Leveled Plans': [
        'Basic Plan',
        'Bronze Plan',
        'Gold Plan',
        'Silver Plan',
        'Premier Plan',  
        'Value Plan',
    ],

    # Group: Annual Plans
    'Annual Plans': [
        'Annual Gold Plan',
        'Annual Silver Plan',
        'Annual Travel Protect Gold',
        'Annual Travel Protect Platinum',
        'Annual Travel Protect Silver',
    ],

    # Group: Miscellaneous Plans
    'Miscellaneous Plans': [
        '24 Protect',
        'Cancellation Plan',        
        'Ticket Protector',
        'Rental Vehicle Excess Insurance',
        'Travel Cruise Protect',
        'Travel Cruise Protect Family',
    ],

    # Group: Individual Plans
    'Individual Plans': [
        'Individual Comprehensive Plan',
        'Single Trip Travel Protect Gold',
        'Single Trip Travel Protect Platinum',
        'Single Trip Travel Protect Silver',
    ],
}

# Encode product for better model processing with numerical fields.
product_encoded = {
    'Annual Plans': 0,
    'Comprehensive Plans': 1,
    'Individual Plans': 2,
    'Leveled Plans': 3,
    'Miscellaneous Plans': 4,
}

## Div for destination
destination_values = ['MALAYSIA', 'AUSTRALIA', 'ITALY', 'UNITED STATES', 'THAILAND',
                      "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF", 'NORWAY', 'VIET NAM', 'DENMARK', 
                      'SINGAPORE', 'JAPAN', 'UNITED KINGDOM', 'INDONESIA', 'INDIA', 'CHINA', 'FRANCE', 
                      'TAIWAN, PROVINCE OF CHINA', 'PHILIPPINES', 'MYANMAR', 'HONG KONG', 
                      'KOREA, REPUBLIC OF', 'UNITED ARAB EMIRATES', 'NAMIBIA', 'NEW ZEALAND', 'COSTA RICA', 
                      'BRUNEI DARUSSALAM', 'POLAND', 'SPAIN', 'CZECH REPUBLIC', 'GERMANY', 'SRI LANKA', 'CAMBODIA', 
                      'AUSTRIA', 'SOUTH AFRICA', 'TANZANIA, UNITED REPUBLIC OF', "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 
                      'NEPAL', 'NETHERLANDS', 'MACAO', 'CROATIA', 'FINLAND', 'CANADA', 'TUNISIA', 'RUSSIAN FEDERATION', 
                      'GREECE', 'BELGIUM', 'IRELAND', 'SWITZERLAND', 'CHILE', 'ISRAEL', 'BANGLADESH', 
                      'ICELAND', 'PORTUGAL', 'ROMANIA', 'KENYA', 'GEORGIA', 'TURKEY', 'SWEDEN', 'MALDIVES',
                      'ESTONIA', 'SAUDI ARABIA', 'PAKISTAN', 'QATAR', 'PERU', 'LUXEMBOURG', 'MONGOLIA', 'ARGENTINA',
                      'CYPRUS', 'FIJI', 'BARBADOS', 'TRINIDAD AND TOBAGO', 'ETHIOPIA', 'PAPUA NEW GUINEA', 'SERBIA',
                      'JORDAN', 'ECUADOR', 'BENIN', 'OMAN', 'BAHRAIN', 'UGANDA', 'BRAZIL', 'MEXICO', 'HUNGARY', 
                      'AZERBAIJAN', 'MOROCCO', 'URUGUAY', 'MAURITIUS', 'JAMAICA', 'KAZAKHSTAN', 'GHANA', 
                      'UZBEKISTAN', 'SLOVENIA', 'KUWAIT', 'GUAM', 'BULGARIA', 'LITHUANIA', 'NEW CALEDONIA', 
                      'EGYPT', 'ARMENIA', 'BOLIVIA', 'VIRGIN ISLANDS, U.S.', 'PANAMA', 'SIERRA LEONE', 'COLOMBIA', 
                      'PUERTO RICO', 'UKRAINE', 'GUINEA', 'GUADELOUPE', 'MOLDOVA, REPUBLIC OF', 'GUYANA', 'LATVIA', 
                      'ZIMBABWE', 'VANUATU', 'VENEZUELA', 'BOTSWANA', 'BERMUDA', 'MALI', 'KYRGYZSTAN', 'CAYMAN ISLANDS', 
                      'MALTA', 'LEBANON', 'REUNION', 'SEYCHELLES', 'ZAMBIA', 'SAMOA', 'NORTHERN MARIANA ISLANDS', 
                      'NIGERIA', 'DOMINICAN REPUBLIC', 'TAJIKISTAN', 'ALBANIA', 
                      'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF', 'LIBYAN ARAB JAMAHIRIYA', 'ANGOLA', 'BELARUS', 
                      'TURKS AND CAICOS ISLANDS', 'FAROE ISLANDS', 'TURKMENISTAN', 'GUINEA-BISSAU', 'CAMEROON', 
                      'BHUTAN', 'RWANDA', 'SOLOMON ISLANDS', 'IRAN, ISLAMIC REPUBLIC OF', 'GUATEMALA', 'FRENCH POLYNESIA',
                      'TIBET', 'SENEGAL', 'REPUBLIC OF MONTENEGRO', 'BOSNIA AND HERZEGOVINA']
destination_options = [{'label': x, 'value': x} for x in destination_values]
input_destination = dcc.Dropdown(
    id='destination',        
    options = destination_options,
    value = 'UNITED STATES'
    )

div_destination = html.Div(
        children=[html.H3('Destination:'), input_destination],
        className="four columns"
        )

#destination encoding
destination_mapping = {
    'MALAYSIA': 'Asia',
    'AUSTRALIA': 'Oceania',
    'ITALY': 'Europe',
    'UNITED STATES': 'North America',
    'THAILAND': 'Asia',
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": 'Asia',
    'NORWAY': 'Europe',
    'VIET NAM': 'Asia',
    'DENMARK': 'Europe',
    'SINGAPORE': 'Asia',
    'JAPAN': 'Asia',
    'UNITED KINGDOM': 'Europe',
    'INDONESIA': 'Asia',
    'INDIA': 'Asia',
    'CHINA': 'Asia',
    'FRANCE': 'Europe',
    'TAIWAN, PROVINCE OF CHINA': 'Asia',
    'PHILIPPINES': 'Asia',
    'MYANMAR': 'Asia',
    'HONG KONG': 'Asia',
    'KOREA, REPUBLIC OF': 'Asia',
    'UNITED ARAB EMIRATES': 'Middle East',
    'NAMIBIA': 'Africa',
    'NEW ZEALAND': 'Oceania',
    'COSTA RICA': 'North America',
    'BRUNEI DARUSSALAM': 'Asia',
    'POLAND': 'Europe',
    'SPAIN': 'Europe',
    'CZECH REPUBLIC': 'Europe',
    'GERMANY': 'Europe',
    'SRI LANKA': 'Asia',
    'CAMBODIA': 'Asia',
    'AUSTRIA': 'Europe',
    'SOUTH AFRICA': 'Africa',
    'TANZANIA, UNITED REPUBLIC OF': 'Africa',
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": 'Asia',
    'NEPAL': 'Asia',
    'NETHERLANDS': 'Europe',
    'MACAO': 'Asia',
    'CROATIA': 'Europe',
    'FINLAND': 'Europe',
    'CANADA': 'North America',
    'TUNISIA': 'Africa',
    'RUSSIAN FEDERATION': 'Europe',
    'GREECE': 'Europe',
    'BELGIUM': 'Europe',
    'IRELAND': 'Europe',
    'SWITZERLAND': 'Europe',
    'CHILE': 'South America',
    'ISRAEL': 'Middle East',
    'BANGLADESH': 'Asia',
    'ICELAND': 'Europe',
    'PORTUGAL': 'Europe',
    'ROMANIA': 'Europe',
    'KENYA': 'Africa',
    'GEORGIA': 'Asia',
    'TURKEY': 'Middle East',
    'SWEDEN': 'Europe',
    'MALDIVES': 'Asia',
    'ESTONIA': 'Europe',
    'SAUDI ARABIA': 'Middle East',
    'PAKISTAN': 'Asia',
    'QATAR': 'Middle East',
    'PERU': 'South America',
    'LUXEMBOURG': 'Europe',
    'MONGOLIA': 'Asia',
    'ARGENTINA': 'South America',
    'CYPRUS': 'Europe',
    'FIJI': 'Oceania',
    'BARBADOS': 'North America',
    'TRINIDAD AND TOBAGO': 'North America',
    'ETHIOPIA': 'Africa',
    'PAPUA NEW GUINEA': 'Oceania',
    'SERBIA': 'Europe',
    'JORDAN': 'Middle East',
    'ECUADOR': 'South America',
    'BENIN': 'Africa',
    'OMAN': 'Middle East',
    'BAHRAIN': 'Middle East',
    'UGANDA': 'Africa',
    'BRAZIL': 'South America',
    'MEXICO': 'North America',
    'HUNGARY': 'Europe',
    'AZERBAIJAN': 'Asia',
    'MOROCCO': 'Africa',
    'URUGUAY': 'South America',
    'MAURITIUS': 'Africa',
    'JAMAICA': 'North America',
    'KAZAKHSTAN': 'Asia',
    'GHANA': 'Africa',
    'UZBEKISTAN': 'Asia',
    'SLOVENIA': 'Europe',
    'KUWAIT': 'Middle East',
    'GUAM': 'Oceania',
    'BULGARIA': 'Europe',
    'LITHUANIA': 'Europe',
    'NEW CALEDONIA': 'Oceania',
    'EGYPT': 'Africa',
    'ARMENIA': 'Asia',
    'BOLIVIA': 'South America',
    'VIRGIN ISLANDS, U.S.': 'North America',
    'PANAMA': 'North America',
    'SIERRA LEONE': 'Africa',
    'COLOMBIA': 'South America',
    'PUERTO RICO': 'North America',
    'UKRAINE': 'Europe',
    'GUINEA': 'Africa',
    'GUADELOUPE': 'North America',
    'MOLDOVA, REPUBLIC OF': 'Europe',
    'GUYANA': 'South America',
    'LATVIA': 'Europe',
    'ZIMBABWE': 'Africa',
    'VANUATU': 'Oceania',
    'VENEZUELA': 'South America',
    'BOTSWANA': 'Africa',
    'BERMUDA': 'North America',
    'MALI': 'Africa',
    'KYRGYZSTAN': 'Asia',
    'CAYMAN ISLANDS': 'North America',
    'MALTA': 'Europe',
    'LEBANON': 'Middle East',
    'REUNION': 'Africa',
    'SEYCHELLES': 'Africa',
    'ZAMBIA': 'Africa',
    'SAMOA': 'Oceania',
    'NORTHERN MARIANA ISLANDS': 'Oceania',
    'NIGERIA': 'Africa',
    'DOMINICAN REPUBLIC': 'North America',
    'TAJIKISTAN': 'Asia',
    'ALBANIA': 'Europe',
    'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF': 'Europe',
    'LIBYAN ARAB JAMAHIRIYA': 'Africa',
    'ANGOLA': 'Africa',
    'BELARUS': 'Europe',
    'TURKS AND CAICOS ISLANDS': 'North America',
    'FAROE ISLANDS': 'Europe',
    'TURKMENISTAN': 'Asia',
    'GUINEA-BISSAU': 'Africa',
    'CAMEROON': 'Africa',
    'BHUTAN': 'Asia',
    'RWANDA': 'Africa',
    'SOLOMON ISLANDS': 'Oceania',
    'IRAN, ISLAMIC REPUBLIC OF': 'Middle East',
    'GUATEMALA': 'North America',
    'FRENCH POLYNESIA': 'Oceania',
    'TIBET': 'Asia',
    'SENEGAL': 'Africa',
    'REPUBLIC OF MONTENEGRO': 'Europe',
    'BOSNIA AND HERZEGOVINA': 'Europe',
}

# Encode regions of destination for better model processing with numerical fields.
region_encoded = {
    'Africa': 0,
    'Asia': 1,
    'Europe': 2,
    'Middle East': 3,
    'North America': 4,
    'Oceania': 5,
    'South America': 6,
}

## Div for agency type
agency_type_values = ['Travel agency', 'Airlines']
agency_type_options = [{'label': x, 'value': x} for x in agency_type_values]
input_agency_type = dcc.Dropdown(
    id='agency_type',
    options=agency_type_options,
    value='Travel agency'
)

agency_encoding = {
    'CBH': 0,
    'CWT': 1,
    'JZI': 2,
    'KML': 3,
    'EPX': 4,
    'C2B': 5,
    'JWT': 6,
    'RAB': 7,
    'SSI': 8,
    'ART': 9,
    'CSR': 10,
    'CCR': 11,
    'ADM': 12,
    'LWC': 13,
    'TTW': 14,
    'TST': 15
}

div_agency_type = html.Div(
    children=[html.H3('Agency Type:'), input_agency_type],
    className="four columns"
)

type_encoding = {
    'Travel agency': 0,
    'Airlines': 1
}

## Div for Distribution Channel
distribution_values = ['Offline','Online']
distribution_options = [{'label': x, 'value': x} for x in distribution_values]
input_distribution = dcc.Dropdown(
    id='distribution',
    options=distribution_options,
    value='Online'
)

div_distribution = html.Div(
    children=[html.H3('Distribution:'), input_distribution],
    className="four columns"
)

dist_encoding = {
    'Online': 0,
    'Offline': 1
}

## Div for numerical characteristics
div_numerical = html.Div(
        children = [div_duration, div_sales, div_age],
        className="row"
        )

## Div for categorical
div_categorical = html.Div(
        children = [div_agency, div_product, div_destination, div_agency_type, div_distribution],
        className="row"
       )

def get_prediction(Agency, Type, Dist_Channel, Product, Destination, Duration, Sales, Age):
        
    cols = ['Agency', 'Type', 'Dist_Channel', 'Product', 'Destination', 'Duration', 'Sales', 'Age', 'Claim']
        
    ## produce a dataframe with a single row of zeros
    df = pd.DataFrame(data = np.zeros((1,len(cols))), columns = cols)    
    
    # get the numeric characteristics
    df.loc[0,'Duration'] = Duration
    df.loc[0,'Sales'] = Sales
    df.loc[0,'Age'] = Age
    
     # Use destination_mapping for encoding 'destination'
    mapped_region = destination_mapping[Destination]
    encoded_region = region_encoded[mapped_region]
    df['Destination'] = encoded_region
    
    # Use product_mapping for encoding 'Product'
    mapped_plan = next((category for category, plans in plan_mapping.items() if Product in plans), None)
    encoded_product = product_encoded[mapped_plan]
    df['Product'] = encoded_product
    
    # encoding for categorial features
    df['Agency'] = agency_encoding[Agency]
    df['Type'] = type_encoding[Type]
    df['Dist_Channel'] = dist_encoding[Dist_Channel]    
      
    # Set 'Claim' to default value of 0 since it will be predicted
    df['Claim'] = 0
        
    #print("DataFrame:")
    #print(df)
    
    # Scale the features using the trained scaler
    df_scaled = scaler.transform(df.drop('Claim', axis=1))
    
    ## Get the predictions using our trained neural network
    prediction = model.predict(df_scaled).flatten()[0]
        
    ## Prediction to binary (claim or no claim) based on a threshold
    threshold = 0.0028038898
    prediction = 1 if prediction >= threshold else 0   
    print('Prediction :', prediction)
    
    # Set 'Claim' to predicted value
    df['Claim'] = prediction
    print(df)
    return prediction

# App layout
app.layout = html.Div([
    html.H1('Travel Insurance Claim Prediction'),

    html.H2('Enter the travel insurance details to predict the claim probability'),

    html.Div(
        children=[div_numerical, div_categorical]
    ),

    html.H1(id='output', style={'margin-top': '50px', 'text-align': 'center'})
])

# Predictor features for the callback
predictors = ['agency', 'agency_type', 'distribution', 'product', 'destination', 'duration', 'sales', 'age']

# Callback function to update the output
@app.callback(
    Output('output', 'children'),
    [Input(x, 'value') for x in predictors])

def show_prediction(Agency, Type, Dist_Channel, Product, Destination, Duration, Sales, Age):
    pred = get_prediction(Agency, Type, Dist_Channel, Product, Destination, Duration, Sales, Age)
    return f"Claim Prediction: {'Yes' if pred == 1 else 'No'}" 

if __name__ == '__main__':
    app.run_server(debug=True)
