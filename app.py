import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.preprocess import scale_input, create_sequences
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile

st.set_page_config(page_title="Human Movement Predictor")

st.title("🧭 Human Movement Predictor")
st.markdown("Upload past movement data to predict the next direction.")

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload movement_data.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.tail())

    model = load_model("model/lstm_model.keras")

    scaled_df, scaler = scale_input(df)
    X, _ = create_sequences(scaled_df, seq_length=5)
    X = np.expand_dims(X[-1], axis=0)

    prediction = model.predict(X)
    prediction = scaler.inverse_transform([prediction[0]])[0]

    # Save to session history
    st.session_state.history.append({
        'last_x': df['x'].iloc[-1],
        'last_y': df['y'].iloc[-1],
        'pred_x': prediction[0],
        'pred_y': prediction[1]
    })

    st.subheader("📍 Predicted Next Location")
    st.write(f"X: {prediction[0]:.2f}, Y: {prediction[1]:.2f}")

    st.subheader("📊 Visual Comparison")
    fig, ax = plt.subplots()
    ax.plot(df['x'], df['y'], label='Past Path')
    ax.scatter(prediction[0], prediction[1], color='red', label='Prediction')
    ax.legend()
    st.pyplot(fig)

    # Save plot as image
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(temp_img.name, format='png')

    # PDF Report Generation
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Human Movement Prediction Report")

    pdf.drawString(50, 750, "📍 Human Movement Prediction Report")
    pdf.drawString(50, 730, f"Predicted Coordinates: X = {prediction[0]:.2f}, Y = {prediction[1]:.2f}")
    pdf.drawString(50, 710, f"Last Known Location: X = {df['x'].iloc[-1]}, Y = {df['y'].iloc[-1]}")
    pdf.drawString(50, 690, f"Total Points: {len(df)}")
    pdf.drawString(50, 670, "Visualization Below:")

    # Embed the plot
    img = ImageReader(temp_img.name)
    pdf.drawImage(img, 50, 400, width=500, height=250)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    st.download_button(
        label="📅 Download PDF Report",
        data=buffer,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )

    st.subheader("🧾 Session Prediction History")
    for idx, item in enumerate(st.session_state.history[::-1]):
        st.markdown(f"**{len(st.session_state.history)-idx}. Last: ({item['last_x']:.2f}, {item['last_y']:.2f}) → Predicted: ({item['pred_x']:.2f}, {item['pred_y']:.2f})**")
