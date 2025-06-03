import streamlit as st
from streamlit_option_menu import option_menu
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
import fitz  # PyMuPDF
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import asyncio

# Inicijalizacija OpenAI klijenta
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o"


# Funkcija za kodiranje slike u base64 format
def encode_image(uploaded_image):
    temp_image_path = "temp_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_image.getvalue())
    with open(temp_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    os.remove(temp_image_path)
    return base64_image


# Funkcija za ekstrakciju teksta iz PDF-a
def extract_text_from_pdf(uploaded_pdf):
    temp_pdf_path = "temp_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.getvalue())
    pdf_document = fitz.open(temp_pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    pdf_document.close()
    os.remove(temp_pdf_path)
    return text


# Funkcija za generisanje slike na osnovu upita
def generate_image_from_prompt(prompt):
    response = requests.post(
        'https://api.openai.com/v1/images/generations',
        headers={
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        },
        json={
            'prompt': prompt,
            'num_images': 1,
            'size': '1024x1024'
        }
    )
    response_data = response.json()
    if 'data' in response_data:
        return response_data['data'][0]['url']
    else:
        st.error("Failed to generate image. Response: " + str(response_data))
        return None


# Funkcija za generisanje grafikona za linearnu regresiju
def generate_linear_regression_chart():
    num_points = st.number_input("Broj tačaka", min_value=2, max_value=100, value=10)
    x = np.random.rand(num_points, 1) * 100
    y = 5 * x + np.random.randn(num_points, 1) * 20

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    plt.scatter(x, y, color='blue', label='Podaci')
    plt.plot(x, y_pred, color='red', linewidth=2, label='Linearni model')
    plt.title("Linearna regresija")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    st.pyplot(plt)


# Funkcija za generisanje grafika funkcije
def generate_function_plot():
    func = st.text_input("Unesite funkciju (npr. np.sin(x), np.cos(x), x**2 + 2*x + 1):")
    if st.button("Generisi grafikon"):
        if func:
            x = np.linspace(-10, 10, 400)
            y = eval(func)

            plt.plot(x, y, label=f'f(x) = {func}')
            plt.title("Grafik funkcije")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            st.pyplot(plt)


# Funkcija za prepoznavanje govora
async def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        st.session_state['status'] = "listening"
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
    except sr.RequestError:
        response = "API error"
    except sr.UnknownValueError:
        response = "Ne mogu da prepoznam govor"
    return response


# Funkcija za pretvaranje teksta u govor
def text_to_speech(text):
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


# Funkcija za dobijanje odgovora od GPT-4
async def get_gpt_response(user_input):
    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_input},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


# Funkcija za generisanje HTML audio elementa
def generate_audio_html(audio_fp):
    audio_bytes = audio_fp.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio id="response-audio" autoplay>
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>
    """
    return audio_html


# HTML i CSS za animaciju balona
animation_html = """
<style>
.bubble {
  width: 200px;
  height: 200px;
  background-color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  margin: auto;
  animation: pulse 2s infinite;
}

.bubble.listening {
  background-color: #ADD8E6;
}

.bubble.responding {
  background-color: #90EE90;
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}
</style>
<div class="bubble" id="bubble">Čekanje...</div>
"""

# Funkcija za ažuriranje statusa balona
def update_bubble(status):
    st.session_state['status'] = status


async def start_listening():
    placeholder = st.empty()
    while not st.session_state['should_stop']:
        user_input = await recognize_speech_from_mic()
        if user_input not in ["API error", "Ne mogu da prepoznam govor"]:
            update_bubble("responding")
            gpt_response = await get_gpt_response(user_input)
            st.session_state['gpt_response_text'] = gpt_response
            audio_response = text_to_speech(gpt_response)
            audio_html = generate_audio_html(audio_response)
            placeholder.markdown(audio_html, unsafe_allow_html=True)
            placeholder2 = st.empty()  # Add a new placeholder for text area
            placeholder2.text_area("GPT-4o Odgovor", st.session_state['gpt_response_text'], height=200, key='response_text')
            await asyncio.sleep(1)
            if st.session_state['should_stop']:
                placeholder.empty()  # Stop voice playback
                placeholder2.empty()  # Clear the text area
                update_bubble("waiting")
                break
            update_bubble("waiting")
        else:
            update_bubble("waiting")
            break


# Glavna Streamlit aplikacija sa opcijama menija
def main():
    with st.sidebar:
        selected = option_menu(
            "Izaberite stranicu",
            ["GPT-4o Chat", "Analiza slike", "Analiza PDF-a", "Generisanje slike", "Generisanje grafikona", "Interact "
                                                                                                            "with "
                                                                                                            "GPT-4o"],
            icons=["chat", "image", "file-pdf", "image", "graph-up", "mic"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "GPT-4o Chat":
        st.title("GPT-4o Chat")
        st.markdown("<style>body { background-color: #EFEFEF; }</style>", unsafe_allow_html=True)

        user_input = st.text_input("Unesite tekst ili postavite pitanje", max_chars=200)
        if st.button("Analiziraj"):
            if user_input:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": user_input}]
                )
                st.write("Assistant:", response.choices[0].message.content)

    elif selected == "Analiza slike":
        st.title("Analiza slike")
        st.markdown("<style>body { background-color: #FFD700; }</style>", unsafe_allow_html=True)

        user_input = st.text_input("Unesite tekst ili postavite pitanje u vezi sa slikom", max_chars=200)
        uploaded_image = st.file_uploader("Izaberite sliku za analizu", type=['png', 'jpg', 'jpeg'])
        if st.button("Analiziraj"):
            if user_input and uploaded_image:
                base64_image = encode_image(uploaded_image)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that can help me with this image."},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"}
                             }
                        ]}
                    ],
                    temperature=0.0,
                )
                st.write("Assistant:", response.choices[0].message.content)

    elif selected == "Analiza PDF-a":
        st.title("Analiza PDF-a")
        st.markdown("<style>body { background-color: #ADD8E6; }</style>", unsafe_allow_html=True)

        uploaded_pdf = st.file_uploader("Izaberite PDF za analizu", type='pdf')
        if uploaded_pdf:
            pdf_text = extract_text_from_pdf(uploaded_pdf)
            st.session_state.pdf_text = pdf_text
            st.success("PDF je uspešno učitan i tekst je ekstraktovan.")

        if 'pdf_text' in st.session_state:
            user_input = st.text_input("Postavite pitanje u vezi sa učitanim PDF-om", max_chars=200)
            if st.button("Analiziraj PDF"):
                if user_input:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": st.session_state.pdf_text},
                                  {"role": "user", "content": user_input}]
                    )
                    st.write("Assistant:", response.choices[0].message.content)

    elif selected == "Generisanje slike":
        st.title("Generisanje slike")
        st.markdown("<style>body { background-color: #90EE90; }</style>", unsafe_allow_html=True)

        user_input = st.text_input("Unesite tekst za generisanje slike", max_chars=200)
        if st.button("Generisi"):
            if user_input:
                image_url = generate_image_from_prompt(user_input)
                if image_url:
                    st.image(image_url, caption='Generisana slika')
                    st.session_state.generated_image_url = image_url

    elif selected == "Generisanje grafikona":
        st.title("Generisanje grafikona")
        chart_type = st.selectbox("Izaberite tip grafikona", ["Linearna regresija", "Grafik funkcije"])
        if chart_type == "Linearna regresija":
            generate_linear_regression_chart()
        elif chart_type == "Grafik funkcije":
            generate_function_plot()

    elif selected == "Interact with GPT-4o":
        st.title("Live Chat sa GPT-4o koristeći mikrofon")

        # Initialize session state
        if 'status' not in st.session_state:
            st.session_state['status'] = "waiting"
        if 'should_stop' not in st.session_state:
            st.session_state['should_stop'] = False
        if 'gpt_response_text' not in st.session_state:
            st.session_state['gpt_response_text'] = ""

        # Display animation
        st.markdown(animation_html, unsafe_allow_html=True)

        # Update bubble class based on status
        bubble_script = f"""
        <script>
          document.getElementById('bubble').className = 'bubble {st.session_state["status"]}';
        </script>
        """
        st.markdown(bubble_script, unsafe_allow_html=True)

        # Control buttons
        if st.button("Pokreni"):
            st.session_state['should_stop'] = False
            asyncio.run(start_listening())

        if st.button("Prekini"):
            st.session_state['should_stop'] = True
            update_bubble("waiting")


if __name__ == "__main__":
    main()
