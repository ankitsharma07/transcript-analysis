import requests
import PyPDF2
import io

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

url = "http://localhost:8000/analyse"

# Open and read the PDF file
with open("./transcripts/abc_call_1.pdf", "rb") as pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)

payload = {"content": pdf_text}

response = requests.post(url, json=payload)

print(response.json())
