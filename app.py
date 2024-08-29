import streamlit as st
from data_processing import load_and_process_data, visualize_data, display_association_results, run_random_forest_regression

# This section is for a custom theme. You can also set this in `.streamlit/config.toml` for global themes.
st.markdown("""
    <style>
        /* You can add more custom styles here */
        .reportview-container {
            background-color: #F0F2F6;
        }
        .sidebar-content {
            background-color: #F0F2F6;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title('📊 Data Visualization and Analysis')

with st.container():
    st.write("""
    ## Upload your CSV file 📁
    Analyze your data with KMeans Clustering , View Association Rules or Random Forest Regresion.
    """)
    
    uploaded_file = st.file_uploader("", type="csv")

    if uploaded_file is not None:
        data, headers, brand_encoder, date_encoder = load_and_process_data(uploaded_file)

        analysis_type = st.radio('Select Analysis Type', ['KMeans Clustering', 'Association Rules', 'Random Forest Regression'])

        if analysis_type == 'KMeans Clustering':
            if st.button("🔍 Visualize Clusters"):
                visualize_data(data, headers, brand_encoder, date_encoder)
        elif analysis_type == 'Association Rules':
            if st.button("🔗 Display Association Results"):
                display_association_results(uploaded_file)
        elif analysis_type == 'Random Forest Regression':
            if st.button("🔗 Run Random Forest Regression"):
                run_random_forest_regression(uploaded_file)
                
