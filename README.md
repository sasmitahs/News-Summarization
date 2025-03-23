# 📰 News Summarization and Text-to-Speech Application

This application fetches news articles about a specified company, summarizes them, performs sentiment analysis, and generates a Hindi audio summary of the final sentiment. Users can input a company name and receive a structured sentiment report along with an audio output.

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
- [Contributors](#-contributors)
- [License](#-license)

## 📖 Introduction

The application extracts and analyzes news articles related to a given company, providing:
1. Summarization of key points.
2. Sentiment analysis (positive, negative, neutral).
3. Comparative sentiment insights across multiple articles.
4. Hindi audio summary of the sentiment.

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
2.Access the application:
Streamlit: Open http://localhost:8501 in your browser.
FastAPI: Open http://localhost:8000 for API endpoints.
🚀 Usage

Enter the company name in the input field (e.g., "Tesla").
Click Fetch News to retrieve and analyze articles.
View the summarized articles, sentiment distribution, and comparative analysis.
Download the JSON report or Hindi audio summary.
🌐 API Details

The backend API is built using FastAPI and provides the following endpoints:

POST /api/fetch_news:
Input: {"company_name": "Tesla"}
Output: JSON containing summarized articles, sentiment analysis, and comparative insights.
POST /api/text_to_speech:
Input: {"company_name": "Tesla"}
Output: Hindi audio file (base64 encoded) and translated text.
GET /api/health:
Output: {"status": "healthy"} (health check endpoint).
🤖 Model Details

The application uses the following models and libraries:

Summarization: T5-small from Hugging Face Transformers.
Sentiment Analysis: Hugging Face's sentiment-analysis pipeline.
Keyword Extraction: KeyBERT for topic modeling.
Hindi Translation: deep-translator with Google Translate API.
Text-to-Speech: gTTS for Hindi audio generation.
⚠️ Assumptions & Limitations

RSS Feeds: The application relies on predefined RSS feeds. If a feed is unavailable or blocked, articles may not be fetched.
Article Limit: Only the first 10 relevant articles are processed.
Language Support: Sentiment analysis and summarization are optimized for English articles.
Network Restrictions: Some feeds may be inaccessible due to network restrictions (e.g., on Hugging Face Spaces).
📊 Expected Output

Articles Section:
Titles, summaries, and sentiment for each article.
Sentiment Distribution:
Metrics for positive, negative, and neutral sentiments.
Comparative Analysis:
Coverage differences and topic overlap across articles.
Hindi Audio:
Downloadable MP3 file with the final sentiment analysis in Hindi.
🚀 Deployment

The application is deployed on Hugging Face Spaces:

Frontend: Streamlit app.
Backend: FastAPI (optional, if hosted separately).
To deploy on Hugging Face Spaces:

Push the code to a GitHub repository.
Create a new Space on Hugging Face and link the repository.
Add the GROQ_API_KEY as a secret environment variable.
Ensure the requirements.txt file includes all dependencies.
