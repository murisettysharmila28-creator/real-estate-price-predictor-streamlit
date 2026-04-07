import streamlit as st
from src.predict import predict_price

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")

st.title("Real Estate Price Prediction App")
st.write("Enter property details to predict the house price.")

year_sold = st.number_input("Year Sold", min_value=1900, max_value=2100, value=2012)
property_tax = st.number_input("Property Tax", min_value=0, value=200)
insurance = st.number_input("Insurance", min_value=0, value=75)
beds = st.number_input("Beds", min_value=0, value=3)
baths = st.number_input("Baths", min_value=0, value=2)
sqft = st.number_input("Square Feet", min_value=100, value=1500)
year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2000)
lot_size = st.number_input("Lot Size", min_value=0, value=5000)
basement = st.selectbox("Basement", [0, 1])
popular = st.selectbox("Popular Area", [0, 1])
recession = st.selectbox("Recession", [0, 1])
property_age = st.number_input("Property Age", min_value=0, value=10)
property_type_condo = st.selectbox("Property Type Condo", [0, 1])

if st.button("Predict Price"):
    input_data = {
        "year_sold": year_sold,
        "property_tax": property_tax,
        "insurance": insurance,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "lot_size": lot_size,
        "basement": basement,
        "popular": popular,
        "recession": recession,
        "property_age": property_age,
        "property_type_Condo": property_type_condo,
    }

    prediction = predict_price(input_data)
    st.success(f"Predicted Property Price: ${prediction:,.2f}")