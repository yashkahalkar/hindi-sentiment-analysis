# 🎭 Hindi Sentiment Analyzer — Text & Audio Based

A robust Hindi Sentiment Analysis system that accepts both **text and audio inputs** and predicts the emotional tone of the speaker as **Happy**, **Sad**, **Angry**, etc.

✨ This project leverages:
- 🎙️ **OpenAI Whisper** for speech-to-text transcription,
- 🧠 **Support Vector Machines (SVM)** for sentiment classification,
- 🔤 **Gemini API** for semantic-rich word embeddings.

---

## 🚀 Features

- 🗣️ **Audio Input Support**  
  Upload a Hindi audio file — transcribed using OpenAI's Whisper and analyzed for sentiment.

- 📝 **Text Input Support**  
  Enter raw Hindi text and get instant sentiment feedback.

- 💡 **Emotion Classification**  
  Detects multiple sentiments such as:
  - 😃 Happy
  - 😞 Sad
  - 😠 Angry
  - 😐 Neutral *(configurable)*

- 🌐 **Multimodal Pipeline**  
  Seamless fusion of audio and text data for natural language understanding.

---

## 🛠️ Tech Stack

| Component       | Tool / Library             |
|----------------|----------------------------|
| Speech-to-Text | [Whisper (OpenAI)](https://github.com/openai/whisper) |
| Embeddings      | Gemini API                 |
| Classifier      | Scikit-learn (SVM)         |
| Language        | Python                     |
| Frontend/UI     | Streamlit *(optional)*     |

---

## 🧩 Model Pipeline

1. **Audio Input** (optional)  
   → Whisper transcribes audio into Hindi text  
2. **Text Preprocessing**  
   → Tokenization, cleaning, etc.  
3. **Embedding Generation**  
   → Gemini API generates contextual embeddings  
4. **Sentiment Classification**  
   → SVM classifies sentiment into predefined categories

---

## 🧪 Sample Usage

### 🔤 Text Input

```python
from model import predict_sentiment

text = "मुझे आज बहुत खुशी हो रही है"
sentiment = predict_sentiment(text)
print("Sentiment:", sentiment)
```

don't forget to star thsi project if you like it ⭐

Made with 💻 by Yash Kahalkar, Email: kahalkaryash@gmail.com 
