# utils.py

import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
import threading
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from keybert import KeyBERT
import queue
from collections import defaultdict
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from groq import Groq
import json
import re

nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load models once
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
sentiment_analyzer = pipeline("sentiment-analysis")
kw_model = KeyBERT()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Initialize Groq client
client = Groq(api_key="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")

# RSS Feeds
rss_feeds = [
    # Technology-focused feeds (general tech news, some may cover Visa tech initiatives)
    "https://feeds.bbci.co.uk/news/technology/rss.xml",  # BBC Technology
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",  # CNBC Tech
    "https://www.theverge.com/rss/index.xml",  # The Verge
    "https://feeds.arstechnica.com/arstechnica/index",  # Ars Technica
    "https://www.engadget.com/rss.xml",  # Engadget
    "https://techcrunch.com/feed/",  # TechCrunch
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",  # NYT Technology
    "https://www.wired.com/feed/rss",  # Wired
    "https://www.zdnet.com/news/rss.xml",  # ZDNet News
    "https://www.cnet.com/rss/news/",  # CNET News
    "https://www.digitaltrends.com/feed/",  # Digital Trends
    "https://www.techmeme.com/feed.xml",  # Techmeme
    "https://www.technologyreview.com/feed/",  # MIT Technology Review
    "https://www.pcworld.com/feed",  # PCWorld
    "https://venturebeat.com/feed/",  # VentureBeat

    # Business and Finance feeds (more likely to cover Visa)
    "https://feeds.bbci.co.uk/news/business/rss.xml",  # BBC Business
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",  # CNBC Business
    "https://www.economist.com/business/rss.xml",  # The Economist Business
    "https://www.ft.com/companies/financials/rss",  # Financial Times Financials (Visa-relevant)
    "https://www.ft.com/rss/companies/technology",  # Financial Times Tech Companies
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",  # Wall Street Journal US Business (updated URL)
    "https://www.forbes.com/money/feed/",  # Forbes Money (updated URL)
    "https://www.reuters.com/arc/outboundfeeds/business/?outputType=xml",  # Reuters Business (updated URL)
    "https://www.bloomberg.com/feed/podcasts/markets.xml",  # Bloomberg Markets (updated URL)
    "https://finance.yahoo.com/news/rssindex",  # Yahoo Finance News
    "https://www.nasdaq.com/feed/rssoutbound",  # Nasdaq News
    "https://www.marketwatch.com/rss/topstories",  # MarketWatch Top Stories
    "https://www.investing.com/rss/news.rss",  # Investing.com News

    # General news (reliable sources that may cover Visa)
    "https://feeds.bbci.co.uk/news/rss.xml",  # BBC News
    "https://www.aljazeera.com/xml/rss/all.xml",  # Al Jazeera
    "https://www.theguardian.com/world/rss",  # The Guardian World
    "https://feeds.npr.org/1001/rss.xml",  # NPR News
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NYT Home Page
    "https://apnews.com/hub/business?format=rss",  # Associated Press Business (updated URL)
    "https://feeds.washingtonpost.com/rss/business",  # Washington Post Business (updated URL)
]


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Locks for thread safety
model_lock = threading.Lock()
sentiment_lock = threading.Lock()
keyword_lock = threading.Lock()

def summarize_t5(text, max_length=100, min_length=30):
    with model_lock:
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def analyze_sentiment(text):
    with sentiment_lock:
        result = sentiment_analyzer(text[:512])[0]
        label = result["label"].lower()
        return "Positive" if label == "positive" else "Negative" if label == "negative" else "Neutral"

def extract_keywords(text):
    with keyword_lock:
        return ", ".join([kw[0] for kw in kw_model.extract_keywords(text, top_n=5)])

def process_article_content(article_data):
    try:
        title, link, content, company_name = article_data
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            summary_future = executor.submit(summarize_t5, content)
            sentiment_future = executor.submit(analyze_sentiment, content)
            keywords_future = executor.submit(extract_keywords, content)
            summary_text = summary_future.result()
            sentiment = sentiment_future.result()
            keywords = keywords_future.result()
        return {
            "title": title,
            "link": link,
            "summary": summary_text,
            "sentiment": sentiment,
            "keywords": keywords
        }
    except Exception as e:
        print(f"❌ Error processing article {title}: {e}")
        return None

def fetch_article_content(article_info, company_name, article_limit_reached):
    title, link, description = article_info
    try:
        if article_limit_reached.is_set():
            return None
        if company_name.lower() in title.lower() or (description and company_name.lower() in description.lower()):
            article_response = requests.get(link, headers=headers, timeout=10)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.content, "html.parser")
            content = "\n".join(p.text for p in article_soup.find_all("p"))
            if company_name.lower() in title.lower() or company_name.lower() in content.lower():
                print(f"✅ Found article: {title}")
                return (title, link, content, company_name)
    except requests.RequestException as e:
        print(f"❌ Failed to retrieve content for: {title} - {e}")
    return None

def fetch_articles_from_rss(rss_url, company_name, article_queue, article_limit_reached):
    try:
        if article_limit_reached.is_set():
            return
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        articles = soup.find_all("item")
        article_infos = [(article.title.text if article.title else "",
                          article.link.text if article.link else "",
                          article.description.text if article.description else "")
                         for article in articles if article.title and article.link]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_article_content, info, company_name, article_limit_reached)
                       for info in article_infos]
            for future in concurrent.futures.as_completed(futures):
                if article_limit_reached.is_set():
                    return
                result = future.result()
                if result:
                    article_queue.put(result)
    except requests.RequestException as e:
        print(f"❌ Failed to fetch RSS feed: {rss_url} - {e}")

def get_coverage_differences(articles, company_name):
    """Fetch coverage differences using Groq API."""
    articles_summary = "\n".join([f"Article {i+1}: Title: {a['title']}, Summary: {a['summary']}, Sentiment: {a['sentiment']}, Keywords: {a['keywords']}"
                                 for i, a in enumerate(articles)])
    prompt = f"""
    Analyze the following ten articles about {company_name} and generate a comparative coverage analysis:
    1. Compare articles based on their main topics.
    2. Identify coverage differences between positive and negative articles.
    3. Provide insights into how these differences impact {company_name}'s market, mentioning article numbers clearly.

    Articles:
    {articles_summary}

    Generate a JSON output in the following format:
    {{
      "Coverage Differences": [
        {{
          "Comparison": "Summary of key differences between two articles.",
          "Impact": "Explanation of how these differences affect {company_name}'s market perception."
        }}
      ]
    }}

    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        coverage_diff = ""
        for chunk in completion:
            coverage_diff += chunk.choices[0].delta.content or ""
        
        text = coverage_diff.strip()  # Fixed: removed space between 'text' and '='
        pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text)
        
        if match:
            json_str = match.group(1)  # Get the content between the markers
            try:
                # Parse the JSON to verify it's valid and return as dictionary
                json_dict = json.loads(json_str)
                json_dict = json.dumps(json_dict, indent=4)
                return json_dict
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format - {str(e)}"
        else:
            return "Error: No JSON content found between ```json and ``` markers"
    except Exception as e:
        return f"Error in Groq API call: {str(e)}"



def similarity_based_common_topics(processed_articles, similarity_threshold=0.8, min_articles=2):
    keyword_clusters = defaultdict(list)
    for article in processed_articles:
        keywords = article["keywords"].split(", ")
        for keyword in keywords:
            if not nlp(keyword).has_vector:
                continue
            added = False
            for cluster_key in list(keyword_clusters.keys()):
                if nlp(keyword).similarity(nlp(cluster_key)) >= similarity_threshold:
                    keyword_clusters[cluster_key].append(keyword)
                    added = True
                    break
            if not added:
                keyword_clusters[keyword].append(keyword)
    deduplicated_clusters = {min(keywords, key=len): keywords for cluster_key, keywords in keyword_clusters.items()}
    common_topics = []
    article_keyword_sets = [set(a["keywords"].split(", ")) for a in processed_articles]
    for representative, cluster in deduplicated_clusters.items():
        articles_with_cluster = sum(1 for keyword_set in article_keyword_sets
                                   if any(kw in keyword_set for kw in cluster))
        if articles_with_cluster >= min_articles:
            common_topics.append(representative)
    final_common_topics = []
    for topic in common_topics:
        if not nlp(topic).has_vector:
            final_common_topics.append(topic)
            continue
        is_similar = False
        for added_topic in list(final_common_topics):
            if nlp(topic).similarity(nlp(added_topic)) >= similarity_threshold:
                is_similar = True
                if len(topic) < len(added_topic):
                    final_common_topics.remove(added_topic)
                    final_common_topics.append(topic)
                break
        if not is_similar:
            final_common_topics.append(topic)
    return final_common_topics

def comparative_analysis(processed_articles, company_name):
    sentiment_summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    all_keywords = []
    for idx, article in enumerate(processed_articles):
        sentiment_summary[article["sentiment"]] += 1
        keywords = set(article["keywords"].split(", "))
        all_keywords.append((idx, keywords))
    common_topics = similarity_based_common_topics(processed_articles)
    unique_topics = {}
    for idx, topics in all_keywords:
        unique = topics - set(common_topics)
        deduplicated_unique = set()
        for topic in unique:
            if not nlp(topic).has_vector:
                deduplicated_unique.add(topic)
                continue
            is_similar = False
            for added_topic in list(deduplicated_unique):
                if nlp(topic).similarity(nlp(added_topic)) >= 0.8:
                    is_similar = True
                    if len(topic) < len(added_topic):
                        deduplicated_unique.remove(added_topic)
                        deduplicated_unique.add(topic)
                    break
            if not is_similar:
                deduplicated_unique.add(topic)
        unique_topics[f"Unique Topics in Article {idx+1}"] = deduplicated_unique
    final_sentiment = max(sentiment_summary, key=sentiment_summary.get)
    # Add stock growth expectation based on sentiment
    if final_sentiment == "Positive":
        sentiment_statement = (f"{company_name}’s latest news coverage is mostly {final_sentiment.lower()}. "
                              f"This positive sentiment suggests potential stock growth as investor confidence may increase.")
    elif final_sentiment == "Negative":
        sentiment_statement = (f"{company_name}’s latest news coverage is mostly {final_sentiment.lower()}. "
                              f"This negative sentiment suggests potential stock decline as investor confidence may weaken.")
    else:  # Neutral
        sentiment_statement = (f"{company_name}’s latest news coverage is mostly {final_sentiment.lower()}. "
                              f"This neutral sentiment suggests limited immediate impact on stock value, with potential for stability unless new developments shift perceptions.")
    
    return {
        "Topic Overlap": {"Common Topics": common_topics, **unique_topics},
        "Final Sentiment Analysis": sentiment_statement
    }

def fetch_and_save_news(company_name):
    if not company_name:
        print("❌ Error: Company name is required")
        return None
    
    file_name = f"{company_name}_news.json"
    articles = []
    article_limit = 10  # Set desired article limit
    article_queue = queue.Queue()
    article_limit_reached = threading.Event()

    print(f"🚀 Starting parallel fetching for {company_name}...")

    # Use all RSS feeds for comprehensive search
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as fetch_executor:
        # Submit all RSS feed fetch tasks
        fetch_futures = [fetch_executor.submit(
            fetch_articles_from_rss, 
            url, 
            company_name,
            article_queue,
            article_limit_reached
        ) for url in rss_feeds]

        # Process articles concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as process_executor:
            processing_futures = []
            
            # Dynamic article processing loop
            while len(articles) < article_limit:
                try:
                    # Get article with timeout
                    article_data = article_queue.get(timeout=2)
                    
                    # Submit for processing
                    future = process_executor.submit(
                        process_article_content, 
                        article_data
                    )
                    processing_futures.append(future)
                    
                except queue.Empty:
                    # Check if we should continue waiting
                    if all(f.done() for f in fetch_futures):
                        print("⚠️ All feeds processed before reaching article limit")
                        break

            # Process completed articles
            for future in concurrent.futures.as_completed(processing_futures):
                result = future.result()
                if result:
                    articles.append(result)
                    print(f"📊 Collected {len(articles)}/{article_limit} articles")
                    
                    # Exit immediately when limit reached
                    if len(articles) >= article_limit:
                        article_limit_reached.set()
                        print(f"✅ Reached {article_limit} articles. Stopping all operations.")
                        break

    # Final article processing
    articles = articles[:article_limit]
    if not articles:
        print(f"❌ No relevant articles found for {company_name}")
        return None
    
    print(f"✅ Saving {len(articles)} articles to {file_name}")
    analysis_result = comparative_analysis(articles, company_name)
    coverage_differences = get_coverage_differences(articles, company_name)
    
    # Parse coverage_differences if it’s a string
    if isinstance(coverage_differences, str):
        try:
            coverage_differences = json.loads(coverage_differences)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Coverage Differences: {e}")
            coverage_differences = {"Coverage Differences": []}
    
    sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        sentiment_distribution[article["sentiment"]] += 1
    
    formatted_articles = [{"Title": article["title"], "Summary": article["summary"],
                           "Sentiment": article["sentiment"], "Topics": article["keywords"].split(", ")}
                          for article in articles]
    
    output_data = {
        "Company": company_name,
        "Articles": formatted_articles,
        "Comparative Sentiment Score": {"Sentiment Distribution": sentiment_distribution},
        "Coverage Differences": coverage_differences,
        "Topic Overlap": {
            "Common Topics": analysis_result['Topic Overlap']['Common Topics'],
            **{k: list(v) for k, v in analysis_result['Topic Overlap'].items() if k != "Common Topics"}
        },
        "Final Sentiment Analysis": analysis_result['Final Sentiment Analysis']
    }
    
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)
    
    print(f"✅ File saved successfully as JSON: {file_name}")
    return file_name

if __name__ == "__main__":
    company_name = input("Enter company name to search for (e.g., Tesla): ")
    fetch_and_save_news(company_name)