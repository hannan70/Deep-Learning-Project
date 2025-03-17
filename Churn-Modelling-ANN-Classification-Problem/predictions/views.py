from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# load all encoder and scaler
model = tf.keras.models.load_model('ml/model.h5', compile=False) 

with open("ml/one_hot_encoder_geography.pkl", 'rb') as file:
  one_hot_encoder_geography = pickle.load(file)

with open("ml/label_encoder_gender.pkl", 'rb') as file:
  label_encoder_gender = pickle.load(file)

with open('ml/scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)



# Create your views here.
def home_page(request):
    if request.method == 'POST':
        credit_score = request.POST.get("CreditScore")
        geography = request.POST.get("Geography")
        gender = request.POST.get("Gender")
        age = request.POST.get("Age")
        tenure = request.POST.get("Tenure")
        balance = request.POST.get("Balance")
        number_of_products = request.POST.get("NumOfProducts")
        hascr_card = request.POST.get("HasCrCard")
        isactive = request.POST.get("IsActiveMember")
        estimated_salary = request.POST.get("EstimatedSalary")

        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': number_of_products,
            'HasCrCard': hascr_card,
            'IsActiveMember': isactive,
            'EstimatedSalary': estimated_salary
        }

        input_df = pd.DataFrame([input_data])

        # apply onehot encoding for geography
        geo_encoder = one_hot_encoder_geography.transform(input_df[['Geography']]).toarray()
        geo_encoder_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

        # Apply label encoding for gender
        input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

        # drop the Geography column
        input_df.drop(columns=['Geography'], axis=1, inplace=True)
        
        dataframe = pd.concat([input_df, geo_encoder_df], axis=1)

        # apply standard scaler
        input_scaled = scaler.transform(dataframe)

        pred = model.predict(input_scaled)
        predict_proba = pred[0][0]

        churm_prob = f"{predict_proba:.2f}"

        return render(request, 'index.html', {"predict_proba": predict_proba, "churm_proba": churm_prob})

    else:
        return render(request, 'index.html')


