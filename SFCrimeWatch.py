import streamlit as st
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import geopy

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.write("Sand Fransisco Crime Watch")

date = st.sidebar.date_input(label="Date:")
time = st.sidebar.time_input(label="Time:")

addressRow1 = st.sidebar.text_input(label="AddressR1")
addressRow2 = st.sidebar.text_input(label="AddressR2")

geolocator = Nominatim(user_agent="GTA Lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geolocator.geocode(addressRow1+" "+addressRow2 + " San Francisco, California")

lat = location.latitude
lon = location.longitude

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})

st.map(map_data) 
st.write("Longitude: " + str(lon) + " | Latitude: " + str(lat))