import pandas as pd
import pickle
import gradio as gr
import numpy as np
from prometheus_client import Gauge, start_http_server, generate_latest
from flask import Flask, Response
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score
import logging
from patient_survival_prediction_model.predict import make_prediction

app = Flask(__name__)

# Gradio app setup
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

def process_input_data(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction,
                 platelets, sex, serum_creatinine, serum_sodium, smoking, time):
    try:
        if None in [age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]: 
            raise ValueError("One or more input values are None. Please provide all input values.")
        # Prepare the input data as a dictionary 
        input_data = { 
            'age': [age], 
            'anaemia': [int(anaemia)], 
            'high_blood_pressure': [int(high_blood_pressure)], 
            'creatinine_phosphokinase': [float(creatinine_phosphokinase)], 
            'diabetes': [int(diabetes)], 
            'ejection_fraction': [float(ejection_fraction)], 
            'platelets': [float(platelets)], 
            'sex': [int(sex)], 
            'serum_creatinine': [float(serum_creatinine)], 
            'serum_sodium': [float(serum_sodium)], 
            'smoking': [int(smoking)], 
            'time': [int(time)] }
        df = pd.DataFrame(input_data)
        df = df.astype({ 
            'age': 'int', 
            'anaemia': 'int', 
            'high_blood_pressure': 'int', 
            'creatinine_phosphokinase': 'float', 
            'diabetes': 'int', 
            'ejection_fraction': 'float', 
            'platelets': 'float', 
            'sex': 'int', 
            'serum_creatinine': 'float', 
            'serum_sodium': 'float', 
            'smoking': 'int', 
            'time': 'int' })
        result = make_prediction(input_data=df)
        predictions_only = result.get("predictions", ["No predictions available"])
        return predictions_only
    except Exception as e:
        logging.error("Error processing input data", exc_info=True)
        return "An error occurred. Please enter valid input data."

iface = gr.Interface(fn = process_input_data,
                        inputs = [ gr.Slider(minimum=0, maximum=100,label="Age"),
                                gr.Radio([0, 1], label="Anaemia (0 for no, 1 for yes)"),
                                gr.Radio([0, 1], label="High Blood Pressure (0 for no, 1 for yes)"),
                                gr.Slider(minimum=0, maximum=10000, label="Creatinine Phosphokinase"),
                                gr.Radio([0, 1], label="Diabetes (0 for no, 1 for yes)"),
                                gr.Slider(minimum=0, maximum=100, label="Ejection Fraction"),
                                gr.Slider(minimum=0, maximum=1000000, label="Platelets"),
                                gr.Radio([0, 1], label="Sex (0 for female, 1 for male)"),
                                gr.Slider(minimum=0, maximum=50, label="Serum Creatinine"),
                                gr.Slider(minimum=0, maximum=200, label="Serum Sodium"),
                                gr.Radio([0, 1], label="Smoking (0 for no, 1 for yes)"),
                                gr.Slider(minimum=0, maximum=365, label="Time (Follow-up period in days)")],
                        outputs = "text",
                        title = title,
                        description = description,
                        allow_flagging='never')

#iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
iface.launch(server_name="127.0.0.1", server_port=8002, share=True)
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8002)