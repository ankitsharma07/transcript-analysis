import uvicorn
import re
from typing import List, Dict, Optional
from collections import  Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import PyPDF2
import io
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["transcript_analysis"]
collection = db["qna"]

app = FastAPI()


class Transcript(BaseModel):
    content: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    count: int
    rating: str


class UpdateQuestionAnswer(BaseModel):
    answer: Optional[str] = None
    count: Optional[int] = None
    rating: Optional[str] = None


def parse_transcript(transcript: str) -> List[tuple]:
    dialogue = []
    current_speaker = ""
    current_text = ""
    current_timestamp = ""
    
    lines = transcript.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if re.match(r'\d{1,2}:\d{2}', line):  # timestamp
            if current_speaker and current_text:
                dialogue.append((current_speaker.strip(), current_text.strip()))
                current_text = ""
            current_timestamp = line
        elif line in ['Nathan', 'Matt']:
            current_speaker = line
        elif line:  
            current_text += " " + line
    
    if current_speaker and current_text:
        dialogue.append((current_speaker.strip(), current_text.strip()))
    
    logger.info(f"Parsed {len(dialogue)} dialogue entries")
    return dialogue


def clean_question(question: str) -> str:
    """Remove pipes and leading/trailing whitespace from a question."""
    return question.replace('|', '').strip()


def extract_seller_questions(dialogue: List[tuple]) -> List[str]:
    seller_questions = []
    for speaker, text in dialogue:
        if speaker.strip() == 'Nathan':
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if '?' in sentence:
                    clean_sentence = clean_question(sentence)
                    if clean_sentence:  # Ensure we're not adding empty strings
                        seller_questions.append(clean_sentence)
    logger.info(f"Extracted {len(seller_questions)} seller questions")
    return seller_questions

def find_similar_questions(questions: List[str]) -> Dict[str, List[str]]:
    if not questions:
        logger.warning("No questions provided to find similar questions")
        return {}
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(questions)
    except ValueError as e:
        logger.error(f"Error in TfidfVectorizer: {str(e)}")
        logger.info(f"Questions: {questions}")
        return {}
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    similar_questions = {}
    for i, question in enumerate(questions):
        similar = [clean_question(questions[j]) for j in range(len(questions)) 
                   if similarity_matrix[i][j] > 0.7 and i != j]
        if similar:
            similar_questions[clean_question(question)] = similar
    
    logger.info(f"Found {len(similar_questions)} questions with similarities")
    return similar_questions


def get_top_questions(similar_questions: Dict[str, List[str]], n: int = 5) -> List[tuple]:
    question_counts = Counter(q for qs in similar_questions.values() for q in qs)
    return question_counts.most_common(n)


def store_questions_answers(top_questions: List[tuple], dialogue: List[tuple]):
    for question, count in top_questions:
        answer = find_answer(question, dialogue)
        rating = rate_answer(answer)
        collection.insert_one({
            "question": question,
            "answer": answer,
            "count": count,
            "rating": rating
        })


def find_answer(question: str, dialogue: List[tuple]) -> str:
    for i, (speaker, text) in enumerate(dialogue):
        if text == question and i+1 < len(dialogue):
            return dialogue[i+1][1]
    return "No answer found!"


def rate_answer(answer: str) -> str:
    if len(answer) > 100:
        return "Best"
    elif len(answer) > 50:
        return "Good"
    else:
        return "Average"


# APIs

@app.post("/analyse")
async def analyse_transcript(transcript: Transcript):
    dialogue = parse_transcript(transcript.content)
    seller_questions = extract_seller_questions(dialogue)
    similar_questions = find_similar_questions(seller_questions)
    
    question_counts = {q: len(similar) for q, similar in similar_questions.items()}
    
    sorted_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_questions = get_top_questions(similar_questions)
    store_questions_answers(top_questions, dialogue)

    return {
        "message": "Analysis complete",
        "similar_questions": similar_questions,
        "question_counts": question_counts,
        "sorted_questions": sorted_questions
    }


@app.get("/questions", response_model=List[QuestionAnswer])
async def get_questions():
    questions = list(collection.find({}, {'_id': 0}))
    return questions


@app.get("/question/{question}", response_model=QuestionAnswer)
async def get_question(question: str):
    result = collection.find_one({'question': question}, {'_id': 0})
    if result:
        return result
    raise HTTPException(status_code=404, detail="Question not found")


@app.put("/question/{question}", response_model=QuestionAnswer)
async def update_question(question: str, update_data: UpdateQuestionAnswer):
    update_dict = update_data.model_dump(exclude_unset=True)
    result = collection.update_one({'question': question}, {'$set': update_dict})
    if result.modified_count:
        updated_question = collection.find_one({'question': question}, {'_id': 0})
        return updated_question
    raise HTTPException(status_code=404, detail="Question not found")


@app.delete("/question/{question}")
async def delete_question(question: str):
    result = collection.delete_one({'question': question})
    if result.deleted_count:
        return {"message": "Question deleted successfully"}
    raise HTTPException(status_code=404, detail="Question not found")


@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    for file in files:
        logger.info(f"Processing file: {file.filename}")
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        if len(pdf_reader.pages) == 0:
            logger.error(f"No pages found in PDF: {file.filename}")
            continue

        transcript = ""
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                transcript += page_text + "\n"
            except Exception as e:
                logger.error(f"Error extracting text from page {i+1} of {file.filename}: {str(e)}")

        if not transcript.strip():
            logger.error(f"No text content extracted from {file.filename}")
            continue

        dialogue = parse_transcript(transcript)
        logger.info(f"Parsed {len(dialogue)} dialogue entries")
        if len(dialogue) == 0:
            logger.warning(f"No dialogue parsed from transcript in file: {file.filename}")
            continue

        seller_questions = extract_seller_questions(dialogue)
        logger.info(f"Extracted {len(seller_questions)} seller questions")
        
        if not seller_questions:
            logger.warning(f"No seller questions found in file: {file.filename}")
            continue
        
        similar_questions = find_similar_questions(seller_questions)
        if not similar_questions:
            logger.warning(f"No similar questions found in file: {file.filename}")
            continue
        
        top_questions = get_top_questions(similar_questions)
        store_questions_answers(top_questions, dialogue)
    
    return {"message": f"Processed {len(files)} PDF files"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
