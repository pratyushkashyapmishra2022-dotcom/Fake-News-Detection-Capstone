import gradio as gr
import joblib

# Load your model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction function
def predict_news(text):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return "Fake" if prediction == 1 else "Real"

# Gradio interface
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=2, placeholder="Enter news headline or article..."),
    outputs="text",
    title="Fake News Detector",
    description="Enter a news headline or short article text, and the model will predict whether it's Fake or Real."
)

if __name__ == "__main__":
    iface.launch()
