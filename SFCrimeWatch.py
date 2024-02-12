import streamlit as st
from streamlit_modal import Modal
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import pandas as pd
import geopy
import pickle
import pickle

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.title("Sand Fransisco Crime Watch")

st.sidebar.title("San Fransisco Crime Watch")
date = st.sidebar.date_input(label="Date:")
time = st.sidebar.time_input(label="Time:")
addressRow1 = st.sidebar.text_input(label="Location")
dayOfWeek = st.sidebar.selectbox("Day of Week", ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
pdDist = st.sidebar.selectbox("Police District:", ('NORTHERN' ,'PARK' ,'INGLESIDE' ,'BAYVIEW' ,'RICHMOND' ,'CENTRAL' ,'TARAVAL'
 ,'TENDERLOIN' ,'MISSION' ,'SOUTHERN'))
pdD_btn = st.sidebar.button(label='Police Department District')
st.sidebar.write("")

#Map of PD District Pop Up
modal = Modal(key="Demo Key",title="Police Department Districts")

#Button for classification
predict_btn = st.sidebar.button("Predict", type='primary')

#Get Longitude and Latitude
geolocator = Nominatim(user_agent="GTA Lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geolocator.geocode(addressRow1+ ", San Francisco, California")
location = geolocator.geocode(addressRow1+ ", San Francisco, California")
lat = location.latitude
lon = location.longitude

#Seperate the dates
month = date.month
dayOfMonth = date.day

#Seperate Time
hour = time.hour
min = time.minute

#Seperate the dates
month = date.month
dayOfMonth = date.day

#Seperate Time
hour = time.hour
min = time.minute

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data) 
st.write("Longitude: " + str(lon) + " | Latitude: " + str(lat))

def Encoder(rawInputDF):
    
    #----DayOfWeek Encoder----
    week_dict = {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
    }
    rawInputDF["DayOfWeek"] = rawInputDF["DayOfWeek"].replace(week_dict)
    
    # #----pdDistrict----
    # cat_dict = {
    #     "LARCENY/THEFT": 0,
    #     "VEHICLE THEFT": 1,
    #     "NON-CRIMINAL": 2,
    #     "ASSAULT": 3,
    #     "DRUG/NARCOTIC": 4
    # }
    # rawInputDF["DayOfWeek"] = rawInputDF["DayOfWeek"].replace(week_dict)
    
    #----pdDistrict----
    pd_dict = {
        "NORTHERN": 0,
        "PARK": 1,
        "INGLESIDE": 2,
        "BAYVIEW": 3,
        "RICHMOND": 4,
        "CENTRAL": 5,
        "TENDERLOIN": 6,
        "TARAVAL": 7,
        "MISSION": 8,
        "SOUTHERN": 9
    }
    rawInputDF["PdDistrict"] = rawInputDF["PdDistrict"].replace(pd_dict)
    
    #---- Standardisation -----
    # Initialize a Scaler
    scaler = MinMaxScaler()
    
    #Scaling X
    OutDF = scaler.fit_transform(rawInputDF)
    OutDF_Scaled = pd.DataFrame(OutDF, columns=['DayOfWeek', 'pdDistrict', 'X', 'Y', 'Month', 'Day', 'Hour', 'Minute'])
    
    return OutDF_Scaled

def TimeFrame(input):
    if(input[6]>=0 and input[6]<4): #Midnight
        daylight = "Midnight"
    elif(input[6]>=4 and input[6]<7): #Early Morning
        daylight = "Early Morning"
    elif(input[6]>=7 and input[6]<12): #Morning
        daylight = "Morning"
    elif(input[6]>=12 and input[6]<17): #Afternoon
        daylight = "Aafternoon"
    elif(input[6]>=17 and input[6]<21): #Evening
        daylight = "Evening"
    elif(input[6]>=21 and input[6]<23): #Night
        daylight = "Night"
        
    return daylight

def crimeType(prediction):    
    if(prediction=="0"): # Larceny/Theft
        crime = "Larceny/Theft"
    elif(prediction=="1"): # Vehicle Theft
        crime = "Vehicle Theft"
    elif(prediction=="2"): # Non-Criminal
        crime = "Non-Criminal"
    elif(prediction=="3"): # Assault
        crime = "Assault"
    elif(prediction=="4"): # Drug/NArcotic
        crime = "Drug/Narcotic"
        
    return crime

if predict_btn:
    #Model Input array
    #Load model
    model = pickle.load(open('finalized_model.sav', 'rb'))
    Input = [dayOfWeek, pdDist, lon, lat, month, dayOfMonth, hour, min]
    InputDF = pd.DataFrame({'DayOfWeek': [dayOfWeek], 'PdDistrict': [pdDist], 'X': [lon], 'Y': [lat], 'Month': [month], 'Day':[dayOfMonth], 'Hour': [hour], 'Minute': [min]})
    ModelInput = Encoder(InputDF)
    
    prediction = model.predict(ModelInput)
    with modal.container():
        st.write("Possible crime: " + str(prediction[0]) + " will occur")

if pdD_btn:
    with modal.container():
        st.image("pdDistrict.png", use_column_width=True)