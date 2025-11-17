import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def rank_resumes(job_description, resumes_texts):
    documents = [job_description] + resumes_texts
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity([job_vector], resume_vectors).flatten()
    return scores

def home(request):
    context = {}
    if request.method == 'POST':
        job_description = request.POST.get('job_description')
        uploaded_files = request.FILES.getlist('resumes')
        resumes_texts, resume_names = [], []

        for uploaded_file in uploaded_files:
            path = default_storage.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(default_storage.location, path)
            text = extract_text_from_pdf(file_path)
            resumes_texts.append(text)
            resume_names.append(uploaded_file.name)

        scores = rank_resumes(job_description, resumes_texts)
        results = pd.DataFrame({'Resume': resume_names, 'Score': scores})
        results = results.sort_values(by='Score', ascending=False)

        context = {
            'results': results.to_dict(orient='records'),
            'top_resume': results.iloc[0]['Resume'],
            'job_description': job_description
        }
    return render(request, 'index.html', context)
