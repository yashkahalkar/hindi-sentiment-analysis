# Hindi Emotion Analysis App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hosted%20on%20Hugging%20Face-yellow)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployed-streamlit-app-link.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI application designed to analyze and predict human emotions from Hindi text and speech. This project leverages state-of-the-art models for speech-to-text, semantic embeddings, and classification, all wrapped in a user-friendly, interactive web interface.

---

### 🌟 Live Demo

**Experience the application live on Streamlit Community Cloud!**

**(▶️ Click the badge above or use this link:[https://hindi-sentiment-analysis-7htwgwjaxuf7anf2qrxmwe.streamlit.app/](https://hindi-sentiment-analysis-7htwgwjaxuf7anf2qrxmwe.streamlit.app/))**

---

### ✨ Features

-   **📝 Text-Based Analysis:** Input Hindi text in Devanagari script to get an instant emotion prediction.
-   **🎤 Speech-Based Analysis:** Upload Hindi audio files (`.wav`, `.mp3`, `.m4a`) to transcribe the speech and analyze its emotional content.
-   **🤖 Multi-Class Classification:** Goes beyond simple positive/negative sentiment to predict a range of emotions such as Joy, Sadness, Anger, Surprise, and more.
-   **💻 Interactive Web Interface:** A clean and intuitive UI built with Streamlit, making the complex backend models accessible to everyone.

---

### 🛠️ Tech Stack & Architecture

This project is built with a modern, decoupled architecture, ensuring scalability and maintainability.

* **Frontend:** [Streamlit](https://streamlit.io/) - For creating the interactive web application.
* **Speech-to-Text (ASR):** [OpenAI Whisper (base model)](https://openai.com/research/whisper) - For highly accurate Hindi audio transcription.
* **Text Embeddings:** [Google Gemini API (`embedding-001`)](https://ai.google.dev/) - To convert text into rich, semantic vector representations.
* **Classifier:** [Scikit-learn](https://scikit-learn.org/) - A Support Vector Machine (SVM) trained for multi-class emotion classification.
* **Model Hosting:** [Hugging Face Hub](https://huggingface.co/) - For storing the trained SVM model artifact, decoupled from the application code.
* **Deployment:** [Streamlit Community Cloud](https://streamlit.io/cloud) - For continuous deployment directly from GitHub.

please give a star if you like it ⭐

Made by Yash Kahalkar 🫡
