import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# Load and prepare the data (original code)
# ---------------------------------------
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

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.title("Shop Type Recommendation System")

st.write("Enter the area details below to predict the best shop type.")

# ---- Numeric Inputs ----
Schools = st.number_input("Schools:", value=0)
Colleges = st.number_input("Colleges:", value=0)
Hospitals = st.number_input("Hospitals:", value=0)
Gyms = st.number_input("Gyms:", value=0)
Boys = st.number_input("Boys_Hostels:", value=0)
Girls = st.number_input("Girls_Hostels:", value=0)
Turfs = st.number_input("Turfs:", value=0)
Parking = st.number_input("Parking_Spaces:", value=0)

Rent = st.number_input("Rent_Cost:", value=0.0)
Footfall = st.number_input("Daily_Footfall:", value=0.0)
Income = st.number_input("Avg_Income:", value=0.0)
Area = st.number_input("Shop_Area_SqFt:", value=0.0)
Transit = st.number_input("Dist_To_Transit:", value=0.0)

# ---- Categorical Inputs ----
Area_Type = st.selectbox("Area Type:", ['Urban', 'Semi-Urban', 'Rural'])
Age_Group = st.selectbox("Dominant Age Group:", ['Kids', 'Teens', 'Adults', 'Seniors'])

# Encode user input the same way as training
area_map = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}
age_map = {'Kids': 1, 'Teens': 3, 'Adults': 0, 'Seniors': 2}

Area_Type_enc = area_map[Area_Type]
Age_Group_enc = age_map[Age_Group]

# -------------------------------------------------------
# Predict Button
# -------------------------------------------------------

if st.button("Predict"):

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

    st.success("Top 3 Recommended Shop Types:")

    for shop, p in zip(top3_classes, top3_probs):
        st.write(f"**{shop}** — {p*100:.2f}% success rate")
