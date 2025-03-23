
# 📰 News Summarization and Text-to-Speech Application

This application fetches news articles about a specified company, summarizes them, performs sentiment analysis, and generates a Hindi audio summary of the final sentiment. Users can input a company name and receive a structured sentiment report along with an audio output.

huggingface spaces link: https://huggingface.co/spaces/Sasmitah/NewsSummarizationTTS1

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Project Setup](#-project-setup)
- [Usage](#-usage)
- [API Details](#-api-details)
- [Model Details](#-model-details)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Expected Output](#-expected-output)
- [Deployment](#-deployment)


## 📖 Introduction

The application extracts and analyzes news articles related to a given company, providing:

- Summarization of key points.
- Sentiment analysis (positive, negative, neutral).
- Comparative sentiment insights across multiple articles.
- Hindi audio summary of the sentiment.

## ✨ Features

- Extracts and displays titles, summaries, and metadata from 10 news articles.
- Sentiment analysis on article content (positive, negative, neutral).
- Comparative sentiment analysis for broader insights.
- Hindi Text-to-Speech (TTS) output.
- Simple, interactive web-based interface using **Streamlit** or **Gradio**.
- API-based architecture for smooth front-end and back-end communication.
- Deployed on **Hugging Face Spaces**.

## 🛠️ Project Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   - **Streamlit**: Open [http://localhost:8501](http://localhost:8501) in your browser.
   - **FastAPI**: Open [http://localhost:8000](http://localhost:8000) for API endpoints.

## 🚀 Usage

- Enter the **company name** in the input field (e.g., "Tesla").
- Click **Fetch News** to retrieve and analyze articles.
- View:
  - Summarized articles with sentiment distribution.
  - Comparative analysis across multiple news articles.
- Download the structured **JSON report** and **Hindi audio summary**.

## 🌐 API Details

The backend API is built using **FastAPI** and provides the following endpoints:

### 1. **POST /api/fetch_news**
- **Input:**
  ```json
  {
    "company_name": "Tesla"
  }
  ```
- **Output:** JSON containing summarized articles, sentiment analysis, and comparative insights.
  
### 2. **POST /api/text_to_speech**
- **Input:**
  ```json
  {
    "company_name": "Tesla"
  }
  ```
- **Output:** Hindi audio file (base64 encoded) and translated text.

### 3. **GET /api/health**
- **Output:**
  ```json
  {
    "status": "healthy"
  }
  ```
- Used for checking the health of the backend service.

### 🧪 API Testing:
You can test the endpoints using **Postman** or **cURL**:

Example:
```bash
curl -X POST http://localhost:8000/api/fetch_news \
     -H "Content-Type: application/json" \
     -d '{"company_name": "Tesla"}'
```

## 🤖 Model Details

The application uses the following models and libraries:

- **Summarization**: `T5-small` model from Hugging Face Transformers for article summarization.
- **Sentiment Analysis**: Hugging Face’s `sentiment-analysis` pipeline for sentiment classification.
- **Keyword Extraction**: `KeyBERT` for extracting key topics from articles.
- **Hindi Translation**: `deep-translator` utilizing the Google Translate API for Hindi translations.
- **Text-to-Speech (TTS)**: `gTTS` (Google Text-to-Speech) for generating Hindi audio outputs.
- - **Coverage Differences**: Utilizes **Groq's llama-3.3-70b-versatile** model for generating comparative insights across news articles.

## ⚠️ Assumptions & Limitations

- **RSS Feeds**:
  - The application relies on predefined RSS feeds.
  - If a feed is unavailable or blocked, articles may not be fetched.
- **Article Limit**:
  - Only the **first 10** relevant articles are processed.
- **Language Support**:
  - Sentiment analysis and summarization are optimized for **English** articles only.
- **Network Restrictions**:
  - Some feeds may be inaccessible due to network restrictions (e.g., on Hugging Face Spaces).

## 📊 Expected Output

1. **Articles Section**:
   - Titles, summaries, and sentiment (positive, negative, neutral) for each article.

2. **Sentiment Distribution**:
   - Metrics showing the count of positive, negative, and neutral sentiments.

3. **Comparative Analysis**:
   - Insights into topic overlap and differences across articles.

4. **Hindi Audio**:
   - Downloadable **MP3** file containing the final sentiment summary in Hindi.


## 🚀 Deployment

The application is deployed on **Hugging Face Spaces**.

### To deploy:

1. **Push the code** to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <repository-url>
   git push -u origin main
   ```

2. **Create a new Space** on Hugging Face and link the repository.


3. **Ensure `requirements.txt`** includes all dependencies.



