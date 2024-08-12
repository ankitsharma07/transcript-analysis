# Running transcript analysis

1. Make sure mongodb is installed and running
2. Assuming that mongodb is running at "mongodb://localhost:27017/"
3. Install the required packages from requirements.txt > pip install -r requirements.txt
4. Also, install one more package called python-multipart > pip install python-multipart
5. To run the project use uvicorn > uvicorn transcript_analysis:app --reload

A local server starts at <http://127.0.0.1>

## Third party libraries

### FastAPI

I have used FastAPI to as a web framework to build the APIs. It is easy to use and provides Swagger documentation for the APIs.

### Pydantic

Data validation and settings management using Python type annotations and it integrates seamlessly with FastAPI.

### scikit-learn [TfIDfVectorizer, Cosine Similarity]

TfidfVectorizer: Converts text to numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency). It helps in quantifying the importance of words in our question

cosine_similarity: Computes similarity between vectors. It is used to find similar questions based on their TF-IDF representations

### PyMongo

To establish a connection with mongodb. It provides python APIs for mongodb.

### PyPDF2

It is used to read and manipulate pdf files.

### uvicorn

ASGI server for running FastAPI apps.

### python-multipart

Streaming multipart parser for python

## APIs

1. `GET /questions`
   - Retrieves all stored questions and answers
   - Response: List of question-answer pairs

2. `GET /question/{question}`
   - Retrieves a specific question and its answer
   - Response: Question-answer pair

3. `PUT /question/{question}`
   - Updates a specific question's data
   - Request body: `{ "answer": "new answer", "count": 5, "rating": "Good" }`
   - Response: Updated question-answer pair

4. `DELETE /question/{question}`
   - Deletes a specific question
   - Response: Deletion confirmation

5. `POST /upload_pdfs`
   - Uploads and processes multiple PDF files
   - Request body: Form data with PDF files
   - Response: Processing confirmation

6. `POST /analyse`
   - Analyzes a given transcript
   - Request body: `{ "content": "transcript text" }`
   - Response: Analysis results including top questions

To run analyse you need to copy all the data of a pdf and put in the content of a request body. I have written a analyse.py script which does it for you. Just give the path of the pdf and make sure the API server is running in the background. To run > python analyse.py

## Methodology

1. PDF text extraction using PyPDF2
2. Dialogue parsing based on timestamps and speaker names (Assuming Nathan is the seller)
3. Seller question extraction using regular expressions
4. Question similarity analysis using TF-IDF and cosine similarity
5. MongoDB for data storage

## Scope of improvement

- The similarity can be improved using word embedding models or BERT model
- Enhancing the parsing method would also help
- We can improve the answer rating system, currently its based on the length of the answer
