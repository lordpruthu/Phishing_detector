import joblib


loaded_model = joblib.load('trained_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

new_input = ["www.google.com",
             "girlhunts.com/online/db9aea5b17fd3967a60fca70695f0116/"]
new_input_tfidf = loaded_vectorizer.transform(new_input)


predictions = loaded_model.predict(new_input_tfidf)


for input_text, prediction in zip(new_input, predictions):
    print(f"Input: {input_text}")
    print(f"Prediction: {'Good' if prediction == 1 else 'Bad'}")
