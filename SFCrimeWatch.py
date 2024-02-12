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

st.title("San Fransisco Crime Watch")

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

def feedback(prediction, input):    
    print(prediction)
    print(input)
    
    if(prediction==0): # Larceny/Theft
        crime = "Larceny/Theft"
        
        if(input[6]>=0 and input[6]<4): #Midnight
            daylight = "Midnight"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Increase Patroling in the area"
            CivMsg = "Highly recommend to stay inside and prevent from being alone outside"
            
        elif(input[6]>=4 and input[6]<7): #Early Morning
            daylight = "Early Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Keep a look out, however msotly safe"
            CivMsg = "Based on previous data, you would be safe but stay alert"
         
        elif(input[6]>=7 and input[6]<12): #Morning
            daylight = "Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Larceny/Theft will start to rise, increase patroling"
            CivMsg = "Based on previous data, stay alert as larceny/theft will start to rise"
         
        elif(input[6]>=12 and input[6]<17): #Afternoon
            daylight = "Afternoon"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Increase patrolling in hotspot areas"
            CivMsg = "Stay alert and make sure belongings are close to your person"
         
        elif(input[6]>=17 and input[6]<21): #Evening
            daylight = "Evening"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Increase patroling and have personel on stadby at hotspots"
            CivMsg = "Highly recommended to stay indoors or walk in a group"
         
        elif(input[6]>=21 and input[6]<23): #Night
            daylight = "Night"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Increase patrols"
            CivMsg = "Highly recommend to stay indoors or walk in a group"
                
    elif(prediction==1): # Vehicle Theft
        crime = "Vehicle Theft"
        
        if(input[6]>=0 and input[6]<4):  #Midnight
            daylight = "Midnight"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium level threat. Based on previous data, keep a lookout"
            CivMsg = "Lock your cars and make sure windows are not wound down"
            
        elif(input[6]>=4 and input[6]<7): #Early Morning
            daylight = "Early Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Low Level threat. Based on previous data, incidents would be the lowest"
            CivMsg = "Your car should be alright, just make sure you have them locked"
         
        elif(input[6]>=7 and input[6]<12): #Morning
            daylight = "Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium level threat. Based on previous data, keep a lookout"
            CivMsg = "If you still have you car parked, just check if its locked. Be safe if you'll be driving!"
         
        elif(input[6]>=12 and input[6]<17): #Afternoon
            daylight = "Afternoon"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium level threat. Based on previous data, incidents will start to rise. Place decoy cars at this time."
            CivMsg = "Relatively safe, make sure cars are locked and handbrakes are up."
         
        elif(input[6]>=17 and input[6]<21): #Evening
            daylight = "Evening"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High level threat. Based on precious data, this is most frequent hours where vehicle theft occurs. Standby near decoy cars."
            CivMsg = "High probability of car theft! Ensure your car is in park and locked. Close all the windows."
         
        elif(input[6]>=21 and input[6]<23): #Night
            daylight = "Night"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High level threat. Patrol and standby near decoy cars."
            CivMsg = "Time frame of high probability of car theft. Lock your cars and ensure the handbrake is up and windows are all the way up."
                    
    elif(prediction==2): # Non-Criminal
        crime = "Non-Criminal"
        
        if(input[6]>=0 and input[6]<4): #Midnight
            daylight = "Midnight"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: Occurences will start to drop."
            CivMsg = ""
            
        elif(input[6]>=4 and input[6]<7): #Early Morning
            daylight = "Early Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Low Occurence: Occurces is currently low but will rise"
            CivMsg = ""
         
        elif(input[6]>=7 and input[6]<12): #Morning
            daylight = "Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium-High Occurence: Be on the lookout for many distress calls as occurence will rise"
            CivMsg = ""
         
        elif(input[6]>=12 and input[6]<17): #Afternoon
            daylight = "Afternoon"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence: Be on standby for many distress calls"
            CivMsg = ""
         
        elif(input[6]>=17 and input[6]<21): #Evening
            daylight = "Evening"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: Occurences will start to drop"
            CivMsg = ""
         
        elif(input[6]>=21 and input[6]<23): #Night
            daylight = "Night"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Low Occurence: Occurences will start to drop"
            CivMsg = ""
                      
    elif(prediction==3): # Assault
        crime = "Assault"
        
        if(input[6]>=0 and input[6]<4): #Midnight
            daylight = "Midnight"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence! Be more vigilant on possible distress calls"
            CivMsg = ""
            
        elif(input[6]>=4 and input[6]<7): #Early Morning
            daylight = "Early Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Low Occurence. Stay on stadby"
            CivMsg = ""
         
        elif(input[6]>=7 and input[6]<12): #Morning
            daylight = "Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence. Warning! Assault occurence will be at a high"
            CivMsg = ""
         
        elif(input[6]>=12 and input[6]<17): #Afternoon
            daylight = "Afternoon"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence! Be on the lookout for potential assault cases and distress calls!"
            CivMsg = ""
         
        elif(input[6]>=17 and input[6]<21): #Evening
            daylight = "Evening"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence! Be on the lookout for potential assault cases and distress calls!"
            CivMsg = ""

        elif(input[6]>=21 and input[6]<23): #Night
            daylight = "Night"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence! Be on the lookout for potential assault cases and distress calls!"
            CivMsg = ""
                      
    elif(prediction==4): # Drug/Narcotic
        crime = "Drug/Narcotic"
        
        if(input[6]>=0 and input[6]<4): #Midnight
            daylight = "Midnight"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: Patrol and keep the lookout for drug/narcotic trades"
            CivMsg = ""
            
        elif(input[6]>=4 and input[6]<7): #Early Morning
            daylight = "Early Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Low Occurence: Eat a donut"
            CivMsg = ""
         
        elif(input[6]>=7 and input[6]<12): #Morning
            daylight = "Morning"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: Drug/Narcotic occurences will rise. Stay vigilant"
            CivMsg = ""
         
        elif(input[6]>=12 and input[6]<17): #Afternoon
            daylight = "Afternoon"
            
            #---|Custom Rule Message|---
            PoliceMsg = "High Occurence: Active drug/narcotic activity will take place! Patrol and stay vigilant."
            CivMsg = ""
         
        elif(input[6]>=17 and input[6]<21): #Evening
            daylight = "Evening"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: drug/narcotic occurence will start to drop"
            CivMsg = ""
         
        elif(input[6]>=21 and input[6]<23): #Night
            daylight = "Night"
            
            #---|Custom Rule Message|---
            PoliceMsg = "Medium Occurence: drug/narcotic occurence will drop further to low occurence."
            CivMsg = ""
            
    crimeMsg = "Crime Category: " + str(crime)          
    dayMsg = "Time of day: " + str(input[6]) + " (" + str(daylight) + ")"
    pdMsg = "Police District: " + str(input[1])
    ContactMsg = "Contact " + input[1] + " if anything were to happen"
    
    feedback = [crimeMsg, dayMsg, pdMsg, PoliceMsg, CivMsg, ContactMsg]
    
    return feedback


if predict_btn:
    #Model Input array
    #Load model
    model = pickle.load(open('finalized_model.sav', 'rb'))
    Input = [dayOfWeek, pdDist, lon, lat, month, dayOfMonth, hour, min]
    InputDF = pd.DataFrame({'DayOfWeek': [dayOfWeek], 'PdDistrict': [pdDist], 'X': [lon], 'Y': [lat], 'Month': [month], 'Day':[dayOfMonth], 'Hour': [hour], 'Minute': [min]})
    ModelInput = Encoder(InputDF)
    
    prediction = model.predict(ModelInput)
    with modal.container():
        # st.write("Possible crime: " + str(prediction[0]) + " will occur")
        fb = feedback(prediction[0], Input)
        st.write(fb[0])
        st.write(fb[1])
        st.write(fb[2])
        st.write("Police : " + fb[3])
        st.write("Civilian : " + fb[4])
        st.write(fb[5])

if pdD_btn:
    with modal.container():
        st.image("pdDistrict.png", use_column_width=True)