from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import time
from api.qa_model import get_answer
from api.summarization_model import summarize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM QA & Summarization API")

class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 200  

class SummarizeRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = 200

request_count = 0

@app.middleware("http")
async def log_requests(request, call_next):
    global request_count
    request_count += 1
    logger.info(f"Request #{request_count} - {request.method} {request.url.path}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response time: {process_time:.2f}s")
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to LLM QA & Summarization API", "endpoints": ["/qa", "/summarize"]}

@app.post("/qa")
def ask_question(request: QuestionRequest):
    try:
        result = get_answer(request.question, max_new_tokens=request.max_tokens)
        logger.info(f"QA completed in {result['response_time']}s")
        return result
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    try:
        result = summarize_text(request.text, max_new_tokens=request.max_tokens)
        logger.info(f"Summarization completed in {result['response_time']}s")
        return result
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
