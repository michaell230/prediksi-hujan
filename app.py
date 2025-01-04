import streamlit as st
import pandas as pd
import numpy as np
from models.random_forest import RandomForest
import pickle
import os

def load_or_train_model(X_train, y_train):
    model_path = 'models/rf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = RandomForest(n_trees=10, min_samples_split=2, max_depth=10)
        model.fit(X_train, y_train)
        # Simpan model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model

def main():
    st.title('Rain Prediction App 🌧️')
    
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('data/processed_rain_data.csv')
        X = df.drop('RainTomorrow', axis=1).values
        y = df['RainTomorrow'].values
        return X, y, df.columns[:-1]  # return feature names juga
    
    X, y, feature_names = load_data()
    
    # Train model jika belum ada
    model = load_or_train_model(X, y)
    
    # Input form
    st.subheader('Enter Weather Parameters')
    
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    
    with col1:
        input_data['MinTemp'] = st.number_input('Minimum Temperature (°C)', value=10.0)
        input_data['MaxTemp'] = st.number_input('Maximum Temperature (°C)', value=20.0)
        input_data['Rainfall'] = st.number_input('Rainfall (mm)', value=0.0)
        input_data['Evaporation'] = st.number_input('Evaporation (mm)', value=5.0)
        input_data['Sunshine'] = st.number_input('Sunshine (hours)', value=8.0)
        input_data['WindGustDir'] = st.selectbox('Wind Gust Direction', 
            options=range(7), 
            format_func=lambda x: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W'][x])
        
    with col2:
        input_data['WindGustspeed'] = st.number_input('Wind Gust Speed (km/h)', value=30.0)
        input_data['WindDir9am'] = st.selectbox('Wind Direction 9am',
            options=range(7),
            format_func=lambda x: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W'][x])
        input_data['WindDir3pm'] = st.selectbox('Wind Direction 3pm',
            options=range(7),
            format_func=lambda x: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W'][x])
        input_data['WindSpeed9am'] = st.number_input('Wind Speed 9am (km/h)', value=10.0)
        input_data['WindSpeed3pm'] = st.number_input('Wind Speed 3pm (km/h)', value=15.0)
        input_data['Humidity9am'] = st.number_input('Humidity 9am (%)', value=70.0)
        input_data['Humidity3pm'] = st.number_input('Humidity 3pm (%)', value=50.0)
        
    with col3:
        input_data['Pressure9am'] = st.number_input('Pressure 9am (hPa)', value=1015.0)
        input_data['Pressure3pm'] = st.number_input('Pressure 3pm (hPa)', value=1015.0)
        input_data['Cloud9am'] = st.number_input('Cloud 9am (oktas)', value=4.0)
        input_data['Cloud3pm'] = st.number_input('Cloud 3pm (oktas)', value=4.0)
        input_data['Temp9am'] = st.number_input('Temperature 9am (°C)', value=15.0)
        input_data['Temp3pm'] = st.number_input('Temperature 3pm (°C)', value=20.0)
        input_data['RainToday'] = st.selectbox('Rain Today', 
            options=[0, 1], 
            format_func=lambda x: 'Yes' if x == 1 else 'No')

    # Prediction button
    if st.button('Predict Rain Tomorrow'):
        # Convert input to array
        input_array = np.array([[
            input_data[feature] for feature in feature_names
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Show prediction
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error('🌧️ It will rain tomorrow!')
            st.markdown("""
            Recommendations:
            - Bring an umbrella
            - Plan indoor activities
            - Check for any outdoor equipment that needs to be protected
            """)
        else:
            st.success('☀️ No rain predicted for tomorrow')
            st.markdown("""
            Recommendations:
            - Good day for outdoor activities
            - Consider watering plants if needed
            - Enjoy the dry weather!
            """)

if __name__ == '__main__':
    main()