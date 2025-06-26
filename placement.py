# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:27:17 2025

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st
from xgboost import XGBRegressor

loaded_model = pickle.load(open('/home/uday/intership sav/placement.sav', 'rb'))

def predict(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return prediction[0]

def main():
    st.title("Salary Prediction App")

    # Numeric inputs
    ssc_p = st.text_input("SSC Percentage")
    hsc_p = st.text_input("HSC Percentage")
    hsc_s = st.text_input("HSC Stream {'Commerce':0, 'Science':1, 'Arts':2}")
    degree_p = st.text_input("Degree Percentage")
    degree_t = st.text_input("Degree Type {'Sci&Tech':0, 'Comm&Mgmt':1, 'Others':2}")
    workex = st.text_input("Work Experience {'No':0, 'Yes':1}")
    etest_p = st.text_input("E-test Percentage")
    specialisation = st.text_input("Specialisation {'Mkt&Fin':0, 'Mkt&HR':1}")
    mba_p = st.text_input("MBA Percentage")
    status = st.text_input("Status {'Placed':0, 'Not Placed':1}")

    if st.button("Predict Salary"):
        try:
            input_list = [
                float(ssc_p), float(hsc_p), float(hsc_s), float(degree_p),
                float(degree_t), float(workex), float(etest_p),
                float(specialisation), float(mba_p), float(status)
            ]
            result = predict(input_list)
            st.success(f"Predicted Salary: ₹ {result:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()