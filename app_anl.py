import streamlit as st
import pandas as pd
import numpy as np
from models.random_forest import RandomForest
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt 
import shap

# processed_rain_data.csv
# predict

def load_model(X_train, y_train): # 
    model_path = 'models/rf_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_feature_importance(model, feature_names, X_test):
    """
    Custom feature importance calculation for our Random Forest model
    """
    n_features = len(feature_names)
    importance_scores = np.zeros(n_features)

    for tree in model.trees:
        def traverse_tree(node, importance):
            if node.feature is not None:  # If not leaf node
                importance[node.feature] += 1
                traverse_tree(node.left, importance)
                traverse_tree(node.right, importance)

        tree_importance = np.zeros(n_features)
        traverse_tree(tree.root, tree_importance)
        importance_scores += tree_importance

        
    # Normalize scores
    importance_scores = importance_scores / np.sum(importance_scores)
    
    # Sort features by importance
    feature_importance = list(zip(feature_names, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    features, scores = zip(*feature_importance)
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(scores)), scores, align='center')
    plt.yticks(range(len(features)), features)
    
    # Add percentage labels
    for idx, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, idx, f' {width:.2%}', 
                va='center', fontweight='bold')
    
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance in Rain Prediction')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return plt

    
def show_analysis(X, y, model, feature_names): 
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm_fig = plot_confusion_matrix(y_test, y_pred)
    st.pyplot(cm_fig)
    
    # Feature Importance
    st.subheader('Feature Importance')
    importance_fig = plot_feature_importance(model, feature_names, X_test)

    # Tambahkan interpretasi
    importance_scores = []
    for tree in model.trees:
        def get_importance(node, scores):
            if node.feature is not None:
                scores[node.feature] += 1
                get_importance(node.left, scores)
                get_importance(node.right, scores)
        
        scores = np.zeros(len(feature_names))
        get_importance(tree.root, scores)
        importance_scores.append(scores)
    
    avg_importance = np.mean(importance_scores, axis=0)
    normalized_importance = avg_importance / np.sum(avg_importance)
    
    # Display top 5 most important features
    st.markdown("#### Top 5 Most Important Features:")
    feature_importance = list(zip(feature_names, normalized_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance[:5]:
        st.write(f"- **{feature}**: {importance:.2%}")
    
    # Display the visualization
    st.pyplot(importance_fig)
    
    # Feature Statistics
    st.subheader('Feature Statistics')
    df = pd.DataFrame(X, columns=feature_names)
    st.dataframe(df.describe())
    
    # Correlation Analysis
    st.subheader('Feature Correlation Analysis')
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    st.pyplot(plt)

def main():
    st.title('Rain Prediction App 🌧️')
    
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv('data/50 50.csv')
        X = df.drop('RainTomorrow', axis=1).values
        y = df['RainTomorrow'].values
        return X, y, df.columns[:-1]
    
    X, y, feature_names = load_data()
    
    # Train model
    model = load_model(X, y)
    
    # Sidebar untuk analisis
    st.sidebar.title('Model Analysis')
    show_analysis_tab = st.sidebar.checkbox('Show Model Analysis')
    
    if show_analysis_tab:
        st.sidebar.subheader('Analysis Options')
        analysis_type = st.sidebar.selectbox(
            'Select Analysis Type',
            ['Model Performance', 'Feature Analysis', 'Model Details']
        )
        
        if analysis_type == 'Model Performance':
            st.header('Model Performance Analysis')
            show_analysis(X, y, model, feature_names)
            
        elif analysis_type == 'Feature Analysis':
            st.header('Feature Analysis')
            st.subheader('Features Used in Model')
            for i, feature in enumerate(feature_names, 1):
                st.write(f"{i}. {feature}")
                
            st.subheader('Feature Importance')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            shap_fig = plot_feature_importance(model, feature_names, X_test)
            st.pyplot(shap_fig)
            
        elif analysis_type == 'Model Details':
            st.header('Model Configuration')
            st.write("""
            ### Random Forest Configuration:
            - Number of Trees: 10
            - Minimum Samples Split: 2
            - Maximum Depth: 10
            - Total Features: {}
            
            ### Model Insights:
            1. Performance Metrics:
               - Model dilatih menggunakan {} sampel
               - Menggunakan bootstrap sampling untuk pembuatan pohon
               - Menerapkan pemilihan fitur secara acak
               
            2. Feature Engineering:
               - Fitur kategorikal telah dienkode
               - Fitur numerik telah dinormalisasi
               - Tidak ada nilai kosong pada data pelatihan
               
            3. Training Process:
               - Menggunakan subset acak fitur untuk setiap split
               - Pohon tumbuh hingga kedalaman maksimum atau sampai node daun menjadi murni
               - Prediksi akhir berdasarkan voting mayoritas
            """.format(len(feature_names), len(X)))
    
    # Original prediction interface
    st.header('Weather Prediction Interface')
    
    input_data = {}
    
    # Kategori: Suhu
    with st.expander("Suhu"):
        input_data['MinTemp'] = st.number_input(
            'Suhu Minimum (°C)', 
            value=10.0, 
            min_value=0.0, 
            max_value=40.0, 
            help="Masukkan suhu minimum dalam rentang 0 hingga 40 °C."
        )
        input_data['MaxTemp'] = st.number_input(
            'Suhu Maksimum (°C)', 
            value=20.0, 
            min_value=0.0, 
            max_value=40.0, 
            help="Masukkan suhu maksimum dalam rentang 0 hingga 40 °C."
        )
        input_data['Temp9am'] = st.number_input(
            'Suhu Pukul 9 Pagi (°C)', 
            value=15.0, 
            help="Masukkan suhu pada pukul 9 pagi dalam °C."
        )
        input_data['Temp3pm'] = st.number_input(
            'Suhu Pukul 3 Sore (°C)', 
            value=20.0, 
            help="Masukkan suhu pada pukul 3 sore dalam °C."
        )

    # Kategori: Data Curah Hujan
    with st.expander("Curah Hujan"):
        input_data['Rainfall'] = st.number_input(
            'Curah Hujan (mm)', 
            value=0.0, 
            min_value=0.0, 
            max_value=40.0, 
            help="Masukkan curah hujan dalam rentang 0 hingga 40 mm."
        )
        input_data['RainToday'] = st.selectbox(
            'Hujan Hari Ini', 
            options=[0, 1], 
            format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
            help="Pilih apakah hujan turun hari ini."
        )

    # Kategori: Data Angin
    with st.expander("Data Angin"):
        input_data['WindGustDir'] = st.selectbox(
            'Arah Angin Kencang', 
            options=range(7), 
            format_func=lambda x: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W'][x],
            help="Pilih arah angin kencang."
        )
        input_data['WindGustspeed'] = st.number_input(
            'Kecepatan Angin Kencang (km/jam)', 
            value=30.0, 
            help="Masukkan kecepatan angin kencang dalam km/jam."
        )
        input_data['WindDir9am'] = st.selectbox(
            'Arah Angin Pukul 9 Pagi',
            options=range(7),
            format_func=lambda x: ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat'][x],
            help="Pilih arah angin pada pukul 9 pagi."
        )
        input_data['WindDir3pm'] = st.selectbox(
            'Arah Angin Pukul 3 Sore',
            options=range(7),
            format_func=lambda x: ['Utara', 'Timur Laut', 'Timur', 'Tenggara', 'Selatan', 'Barat Daya', 'Barat'][x],
            help="Pilih arah angin pada pukul 3 sore."
        )
        input_data['WindSpeed9am'] = st.number_input(
            'Kecepatan Angin Pukul 9 Pagi (km/jam)', 
            value=10.0, 
            help="Masukkan kecepatan angin pada pukul 9 pagi."
        )
        input_data['WindSpeed3pm'] = st.number_input(
            'Kecepatan Angin Pukul 3 Sore (km/jam)', 
            value=15.0, 
            help="Masukkan kecepatan angin pada pukul 3 sore."
        )
        input_data['Pressure9am'] = st.number_input(
            'Tekanan Udara Pukul 9 Pagi (hPa)', 
            value=1015.0, 
            help="Masukkan tekanan udara pada pukul 9 pagi dalam hPa."
        )
        input_data['Pressure3pm'] = st.number_input(
            'Tekanan Udara Pukul 3 Sore (hPa)', 
            value=1015.0, 
            help="Masukkan tekanan udara pada pukul 3 sore dalam hPa."
        )

    # Kategori: Data Lainnya
    with st.expander("Data Lainnya"):
        input_data['Evaporation'] = st.number_input(
            'Penguapan (mm)', 
            value=5.0, 
            min_value=0.0, 
            max_value=40.0, 
            help="Masukkan penguapan dalam rentang 0 hingga 40 mm."
        )
        input_data['Sunshine'] = st.number_input(
            'Cahaya Matahari (jam)', 
            value=8.0, 
            min_value=0.0, 
            max_value=40.0, 
            help="Masukkan jumlah cahaya matahari dalam rentang 0 hingga 40 jam."
        )
        input_data['Humidity9am'] = st.number_input(
            'Kelembapan Pukul 9 Pagi (%)', 
            value=70.0, 
            help="Masukkan kelembapan pada pukul 9 pagi dalam persen."
        )
        input_data['Humidity3pm'] = st.number_input(
            'Kelembapan Pukul 3 Sore (%)', 
            value=50.0, 
            help="Masukkan kelembapan pada pukul 3 sore dalam persen."
        )
        input_data['Cloud9am'] = st.number_input(
            'Awan Pukul 9 Pagi (okta)', 
            value=4.0, 
            help="Masukkan jumlah awan pada pukul 9 pagi dalam okta."
        )
        input_data['Cloud3pm'] = st.number_input(
            'Awan Pukul 3 Sore (okta)', 
            value=4.0, 
            help="Masukkan jumlah awan pada pukul 3 sore dalam okta."
        )

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
            st.success('🌧️ It will rain tomorrow!')
            st.markdown("""
            Recommendations:
            - Bring an umbrella
            - Plan indoor activities
            - Check for any outdoor equipment that needs to be protected
            """)
        else:
            st.error('☀️ No rain predicted for tomorrow')
            st.markdown("""
            Recommendations:
            - Good day for outdoor activities
            - Consider watering plants if needed
            - Enjoy the dry weather!
            """)

if __name__ == '__main__':
    main()
