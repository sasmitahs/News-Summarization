from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import io
import json
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Analysis API")
translator = GoogleTranslator(source='en', target='hi')

class CompanyRequest(BaseModel):
    company_name: str

@app.post("/api/fetch_news")
async def fetch_news(request: CompanyRequest):
    try:
        company_name = request.company_name.strip().lower()
        if not company_name:
            raise HTTPException(status_code=400, detail="Company name is required")
        
        logger.info(f"Fetching news for {company_name}")
        file_name = utils.fetch_and_save_news(company_name)
        if not file_name:
            logger.warning(f"No news found for {company_name}")
            raise HTTPException(status_code=404, detail=f"No news found for {company_name}")
        
        with open(file_name, "r", encoding="utf-8") as file:
            content = file.read()
        
        try:
            news_data = json.loads(content)  # Should work with updated utils.py
            logger.info(f"Successfully parsed news data for {company_name}")
            return news_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error parsing JSON: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in fetch_news: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.post("/api/text_to_speech")
async def text_to_speech(request: CompanyRequest):
    try:
        company_name = request.company_name.strip().lower()
        if not company_name:
            raise HTTPException(status_code=400, detail="Company name is required")
        
        file_name = f"{company_name}_news.txt"
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                news_data = json.load(file)
            sentiment_text = news_data.get("Final Sentiment Analysis", "")
            if not sentiment_text:
                raise HTTPException(status_code=404, detail="Sentiment analysis not found")
            
            hindi_text = translator.translate(sentiment_text)
            tts = gTTS(text=hindi_text, lang='hi')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
            return {"text": hindi_text, "audio_base64": audio_base64}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"News file for {company_name} not found")
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)