import joblib
import spacy

initial = "California chain serves up best fast-food burger in nation, USA Today says. It\u2019s not In-N-Out [The Habit Burger Grill]"

nlp = spacy.load("en_core_web_sm")

def extract_location(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]

location = extract_location(initial)
print(f"Locations: {location}")

model_pipe = joblib.load('disaster_model.pkl')
prediction = model_pipe.predict([initial])
probability = model_pipe.predict_proba([initial])

print(f"Prediction: {prediction[0]}")
print(f"Confidence: {probability[0][prediction[0]]}")