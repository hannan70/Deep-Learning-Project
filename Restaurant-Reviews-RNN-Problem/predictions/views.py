from django.shortcuts import render
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences
import re


# load model
from tensorflow.keras.models import load_model

model = load_model('ml/simple_rnn.h5', compile=False)


# Create your views here.
def home_page(request):
    

    if request.method == 'POST':
        input_review = request.POST.get("review")

        voc_size=10000
        max_length=500
        # Function to preprocess user input
        def preprocess_text(text):
            reviews = re.sub('[^a-zA-Z]', ' ', text)
            reviews = reviews.lower() 
            encoded_review = one_hot(reviews, voc_size)
            padded_review = pad_sequences([encoded_review], maxlen=max_length)
            return padded_review
        

        def predict_sentiment(review):
            processes_input = preprocess_text(review)
            prediction = model.predict(processes_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            return sentiment, prediction[0][0]
    

        sentiment, score = predict_sentiment(input_review)
        score_percentage  = f"{score:.2f}%"
        return render(request, "index.html", {"sentiment": sentiment, "score": score_percentage })
    else:
        return render(request, "index.html")