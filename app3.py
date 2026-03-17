import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Shop Recommendation System",
    page_icon="🏪",
    layout="wide"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 70px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 2rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Load and prepare the data
# ---------------------------------------
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Shop_rec_final.csv")
    df.drop("Competitors", inplace=True, axis=1)

    # Fill missing values
    cols_cat = ['Schools','Colleges','Hospitals','Gyms','Boys_Hostels','Girls_Hostels','Turfs','Parking_Spaces']
    for col in cols_cat:
        df[col].fillna(df[col].mode()[0], inplace=True)

    cols_num = ['Rent_Cost','Daily_Footfall','Avg_Income','Shop_Area_SqFt','Dist_To_Transit']
    for col in cols_num:
        df[col].fillna(df[col].mean(), inplace=True)

    df['Area_Type'].fillna(df['Area_Type'].mode()[0], inplace=True)
    df['Dominant_Age_Group'].fillna(df['Dominant_Age_Group'].mode()[0], inplace=True)

    # Encoding
    lab = LabelEncoder()
    df['Area_Type'] = lab.fit_transform(df['Area_Type'])
    df['Dominant_Age_Group'] = lab.fit_transform(df['Dominant_Age_Group'])

    # Independent and dependent variables
    x = df.drop(['Best_Shop_Type'], axis=1)
    y = df['Best_Shop_Type']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Model
    rfc = RandomForestClassifier(n_estimators=100, random_state=2)
    rfc.fit(x_train, y_train)
    
    return rfc, scaler

rfc, scaler = load_and_train_model()

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

# Header
st.markdown('<div class="main-header">🏪 Shop Type Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get AI-powered recommendations for the best shop type based on location analytics</p>', unsafe_allow_html=True)

st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">📍 Location Amenities</p>', unsafe_allow_html=True)
    
    Schools = st.number_input("🏫 Schools", value=0, min_value=0, help="Number of schools nearby")
    Colleges = st.number_input("🎓 Colleges", value=0, min_value=0, help="Number of colleges nearby")
    Hospitals = st.number_input("🏥 Hospitals", value=0, min_value=0, help="Number of hospitals nearby")
    Gyms = st.number_input("💪 Gyms", value=0, min_value=0, help="Number of gyms nearby")
    Boys = st.number_input("🏠 Boys Hostels", value=0, min_value=0, help="Number of boys hostels nearby")
    Girls = st.number_input("🏡 Girls Hostels", value=0, min_value=0, help="Number of girls hostels nearby")
    Turfs = st.number_input("⚽ Turfs", value=0, min_value=0, help="Number of sports turfs nearby")
    Parking = st.number_input("🅿️ Parking Spaces", value=0, min_value=0, help="Number of parking spaces available")

with col2:
    st.markdown('<p class="section-header">📊 Area Characteristics</p>', unsafe_allow_html=True)
    
    Rent = st.number_input("💰 Rent Cost (₹)", value=0.0, min_value=0.0, help="Monthly rent cost")
    Footfall = st.number_input("👥 Daily Footfall", value=0.0, min_value=0.0, help="Average daily footfall")
    Income = st.number_input("💵 Average Income (₹)", value=0.0, min_value=0.0, help="Average income of residents")
    Area = st.number_input("📐 Shop Area (Sq Ft)", value=0.0, min_value=0.0, help="Shop area in square feet")
    Transit = st.number_input("🚉 Distance to Transit (km)", value=0.0, min_value=0.0, help="Distance to nearest transit")
    
    st.markdown('<p class="section-header">🏘️ Demographics</p>', unsafe_allow_html=True)
    
    Area_Type = st.selectbox("🌆 Area Type", ['Urban', 'Semi-Urban', 'Rural'], help="Type of area")
    Age_Group = st.selectbox("👨‍👩‍👧‍👦 Dominant Age Group", ['Kids', 'Teens', 'Adults', 'Seniors'], help="Primary age group in area")

# Encode user input
area_map = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}
age_map = {'Kids': 1, 'Teens': 3, 'Adults': 0, 'Seniors': 2}

Area_Type_enc = area_map[Area_Type]
Age_Group_enc = age_map[Age_Group]

# -------------------------------------------------------
# Predict Button
# -------------------------------------------------------

if st.button("🔮 Get Recommendations"):
    with st.spinner("Analyzing location data..."):
        input_data = [[
            Schools, Colleges, Hospitals, Gyms, Boys, Girls,
            Turfs, Parking,
            Rent, Footfall, Income, Area, Transit,
            Area_Type_enc, Age_Group_enc
        ]]

        input_scaled = scaler.transform(input_data)

        # Top 3 predictions
        probs = rfc.predict_proba(input_scaled)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        classes = rfc.classes_
        top3_classes = classes[top3_idx]
        top3_probs = probs[top3_idx]

        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Top 3 Recommended Shop Types</p>', unsafe_allow_html=True)

        # Display results with colored boxes
        for i, (shop, p) in enumerate(zip(top3_classes, top3_probs)):
            if i == 0:
                color = "#2ecc71"
                medal = "🥇"
            elif i == 1:
                color = "#3498db"
                medal = "🥈"
            else:
                color = "#e67e22"
                medal = "🥉"
            
            st.markdown(f"""
                <div style="background-color: {color}; padding: 0.8rem; max-width:420px; margin: 0 auto 0.6rem auto; border-radius: 8px; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                    <h3 style="margin: 0; color: white;">{medal} {shop}</h3>
                    <p style="margin-top: 0.3rem; color: white;">Success Rate: {p*100:.2f}%</p>
                    <div style="background-color: rgba(255,255,255,0.3); height: 6px; border-radius: 5px; margin-top: 0.5rem;">
                        <div style="background-color: white; height: 100%; width: {p*100}%; border-radius: 5px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>💡 Powered by Machine Learning | Random Forest Classifier</p>
    </div>
""", unsafe_allow_html=True)