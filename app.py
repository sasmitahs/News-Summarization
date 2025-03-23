import streamlit as st
import requests
import json
import base64
import io

API_BASE_URL = "http://localhost:8000/api"

st.title("News Summarization and Text-to-Speech Application")

company_name = st.text_input("Enter the company name:", "").strip().lower()

if st.button("Fetch News"):
    if company_name:
        status = st.status("Fetching news...", expanded=True)
        status.write(f"Fetching news for **{company_name}**...")
        try:
            response = requests.post(
                f"{API_BASE_URL}/fetch_news",
                json={"company_name": company_name},
                timeout=120
            )
            response.raise_for_status()
            
            news_data = response.json()
            if not news_data or "Company" not in news_data:
                status.update(label="No news found", state="error")
                st.warning(f"No news found for {company_name}")
            else:
                status.update(label="News fetched successfully!", state="complete", expanded=False)
                
                st.subheader(f"News Analysis for {news_data['Company']}")
                
                # Articles section
                st.subheader("Articles")
                with st.expander("View Articles", expanded=False):
                    for i, article in enumerate(news_data['Articles']):
                        st.markdown(f"#### Article {i+1}: {article['Title']}")
                        st.markdown(f"**Summary:** {article['Summary']}")
                        st.markdown(f"**Sentiment:** {article['Sentiment']}")
                        st.markdown(f"**Topics:** {', '.join(article['Topics'])}")
                        st.divider()
                
                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                sentiment_data = news_data['Comparative Sentiment Score']['Sentiment Distribution']
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", sentiment_data['Positive'])
                col2.metric("Neutral", sentiment_data['Neutral'])
                col3.metric("Negative", sentiment_data['Negative'])
                
                # Topic Analysis
                st.subheader("Topic Analysis")
                with st.expander("View Topic Analysis", expanded=False):
                    st.markdown("**Common Topics:**")
                    st.write(", ".join(news_data['Topic Overlap']['Common Topics']))
                    for key, value in news_data['Topic Overlap'].items():
                        if key != "Common Topics":
                            st.markdown(f"**{key}:**")
                            st.write(", ".join(value))
                
                # Coverage Differences
                st.subheader("Coverage Differences")
                with st.expander("View Comparative Analysis", expanded=False):
                    coverage_diff = news_data['Coverage Differences']
                    if isinstance(coverage_diff, str):
                        st.write(coverage_diff)  # Fallback for error cases
                    else:
                        # Format line-by-line
                        formatted_text = '"Coverage Differences": [\n'
                        for i, item in enumerate(coverage_diff.get("Coverage Differences", [])):
                            formatted_text += "{\n"
                            formatted_text += f'    "Comparison": "{item["Comparison"]}",\n'
                            formatted_text += f'    "Impact": "{item["Impact"]}"\n'
                            formatted_text += "}" + (",\n" if i < len(coverage_diff["Coverage Differences"]) - 1 else "\n")
                        formatted_text += "]"
                        st.code(formatted_text, language="json")
                
                # Final Sentiment Analysis
                st.subheader("Final Sentiment Analysis")
                st.info(news_data['Final Sentiment Analysis'])
                
                # Download JSON
                st.subheader("Download Data")
                st.download_button(
                    label="Download JSON File",
                    data=json.dumps(news_data, indent=4),
                    file_name=f"{company_name}_news.json",
                    mime="application/json"
                )
                
                # Hindi Audio
                st.subheader("Hindi Audio for Final Sentiment Analysis")
                audio_response = requests.post(
                    f"{API_BASE_URL}/text_to_speech",
                    json={"company_name": company_name},
                    timeout=60
                )
                audio_response.raise_for_status()
                audio_data = audio_response.json()
                #st.markdown(f"**Hindi translation:**")
                #st.text(audio_data["text"])
                audio_bytes = base64.b64decode(audio_data["audio_base64"])
                #st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    label="Download Hindi Audio",
                    data=audio_bytes,
                    file_name=f"{company_name}_sentiment_hindi.mp3",
                    mime="audio/mp3"
                )
        
        except requests.exceptions.RequestException as e:
            status.update(label="Connection error", state="error")
            st.error(f"Error connecting to API: {str(e)}")
            st.info("Make sure the FastAPI backend is running on http://localhost:8000")
        except json.JSONDecodeError:
            status.update(label="Invalid response", state="error")
            st.error("Received invalid data from the API")
        except Exception as e:
            status.update(label="Processing error", state="error")
            st.error(f"Error processing news data: {str(e)}")
    else:
        st.warning("Please enter a company name.")