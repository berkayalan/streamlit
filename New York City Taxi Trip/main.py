import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from PIL import Image

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

header = st.container()
dataset = st.container()
features = st.container()
model_trainging = st.container()

st.markdown(
    """
    <style>
    .main {background-color: #F5F5F5};
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    train_data = pd.read_csv(filename)
    return train_data


with header:    
    st.title("Welcome to this Kaggle Data Science Project")
    st.text("""Share code and data to improve ride time predictions""")

with dataset:
    st.header("New York City Taxi Trip Duration")
    #https://docs.streamlit.io/en/stable/api.html#display-media
    image = Image.open('taxi.png')
    st.image(image, caption='Main Image of Competition')

    st.text("""
    Dataset is taken from Kaggle.\nLink is here: https://www.kaggle.com/c/nyc-taxi-trip-duration/data?select=test.zip
    
    The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform.\n The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition.\n Based on individual trip attributes, participants should predict the duration of each trip in the test set.
    """)    

    st.subheader("Let's Look at the train data!")
    train_data = get_data("data/train.csv")
    st.write(train_data.head(10))

    st.subheader("Columns")
    st.text("""
    
    id - a unique identifier for each trip

    vendor_id - a code indicating the provider associated with the trip record
    
    pickup_datetime - date and time when the meter was engaged
    
    dropoff_datetime - date and time when the meter was disengaged
    
    passenger_count - the number of passengers in the vehicle (driver entered value)
    
    pickup_longitude - the longitude where the meter was engaged
    
    pickup_latitude - the latitude where the meter was engaged
    
    dropoff_longitude - the longitude where the meter was disengaged
    
    dropoff_latitude - the latitude where the meter was disengaged
    
    store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
    
    trip_duration - duration of the trip in seconds

    """)


    st.subheader("Let's Look at the occurence of Passenger Counts")
    pdist= pd.DataFrame(train_data["passenger_count"].value_counts())
    st.bar_chart(pdist)
    #https://docs.streamlit.io/en/stable/api.html#display-charts


with features:
    st.header("Features Created by Me")
    st.markdown("* **First Feature*: **        \nI calculated it with this logic: "  )
    st.markdown("* **Second Feature*: **        \nI calculated it with this logic: "  )
    

    st.subheader("Distance Vs Trip Duration(Haversine Distance)")
    st.text("""
    Based off exploratory data analysis on kaggle, the distance (km) between pickup and dropoff points is a significant feature impacting trip duration.\n Let's calculate the distance and investigate its patterns by using haversine formula.
    
    """)    


    
with model_trainging:
    st.header("Training")
    st.text("Training Process of the model")

    sel,display = st.columns(2)
    n_estimators=sel.selectbox("What should be tree size of the model?",options=[10,100,200,500,1000,"No Limit"],index=1)
    max_depth=sel.slider("What should be max_depth size of the model?",min_value=10,max_value=100,value=20,step=10)
    min_samples_leaf=sel.slider("What should be min_samples_leaf size of the model?",min_value=1,max_value=10,value=3,step=1)

    selected_feature= sel.selectbox("Which feature should be used as input feature?",options =train_data.columns,index=4)
    
    X= train_data[[selected_feature]]
    y= train_data[["trip_duration"]]

    if n_estimators=="No Limit":
        random_forests = RandomForestRegressor(random_state=60, max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    else:
        random_forests = RandomForestRegressor(random_state=60,n_estimators=n_estimators,
                                            max_depth=max_depth,min_samples_leaf=min_samples_leaf)

    random_forests.fit(X,y)
    prediction = random_forests.predict(y)

    display.subheader("Mean absolute error of the model is: ")
    display.write(mean_absolute_error(y,prediction))
    display.subheader("Mean squared error of the model is: ")
    display.write(mean_squared_error(y,prediction))
    display.subheader("R squred score of the model is: ")
    display.write(r2_score(y,prediction))





