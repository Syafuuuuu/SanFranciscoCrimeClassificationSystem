import streamlit as st
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import geopy
import pickle

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.write("Sand Fransisco Crime Watch")

date = st.sidebar.date_input(label="Date:")
time = st.sidebar.time_input(label="Time:")

addressRow1 = st.sidebar.text_input(label="Location")
dayOfWeek = st.sidebar.selectbox("Day of Week", ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
pdDist = st.sidebar.selectbox("Police District:", ('NORTHERN' ,'PARK' ,'INGLESIDE' ,'BAYVIEW' ,'RICHMOND' ,'CENTRAL' ,'TARAVAL'
 ,'TENDERLOIN' ,'MISSION' ,'SOUTHERN'))

#Button for classification
predict_btn = st.sidebar.button("Predict", type='primary')

#Get Longitude and Latitude
geolocator = Nominatim(user_agent="GTA Lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geolocator.geocode(addressRow1+ ", San Francisco, California")
lat = location.latitude
lon = location.longitude

#Seperate the dates
month = date.month
dayOfMonth = date.day

#Seperate Time
hour = time.hour
min = time.minute

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data) 
st.write("Longitude: " + str(lon) + " | Latitude: " + str(lat))

def hotEncode(rawInputDF):
    
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
    
    
    
    return InputDF

if predict_btn:
    #Model Input array
    Input = [dayOfWeek, pdDist, lon, lat, month, dayOfMonth, hour, min]
    InputDF = pd.DataFrame({'DayOfWeek': [dayOfWeek], 'pdDistrict': [pdDist], 'X': [lon], 'Y': [lat], 'Month': [month], 'Day':[dayOfMonth], 'Hour': [hour], 'Minute': [min]})
    print(InputDF)
    # st.write(Input)
    