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

- Extracts and displays titles, summaries, and metadata from 10+ news articles.
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
