import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from urllib.parse import quote_plus

SKILL_KEYWORDS = [
    'python', 'java', 'c', 'c++', 'c#', 'javascript', 'typescript', 'react', 'angular', 'vue',
    'node', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'dotnet', '.net',
    'html', 'css', 'sass', 'tailwind', 'bootstrap',
    'sql', 'postgresql', 'mysql', 'sqlite', 'oracle', 'mssql',
    'nosql', 'mongodb', 'dynamodb', 'cassandra', 'redis',
    'graphql', 'rest', 'api',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
    'linux', 'bash', 'powershell',
    'git', 'github', 'gitlab', 'bitbucket',
    'jira', 'confluence',
    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn',
    'machine learning', 'deep learning', 'nlp', 'computer vision',
    'spark', 'hadoop', 'airflow', 'kafka',
    'tableau', 'power bi', 'excel'
]

def _normalize_text(s):
    return re.sub(r"[^a-z0-9\+\.# ]+", " ", s.lower())

def extract_skills(text, keywords):
    t = _normalize_text(text)
    hay = f" {t} "
    found = set()
    for kw in keywords:
        k = _normalize_text(kw)
        if f" {k} " in hay:
            found.add(kw)
    return found

def classify_match_label(pct):
    if pct >= 85:
        return 'Excellent Match', 'score-excellent'
    if pct >= 70:
        return 'Good Match', 'score-good'
    if pct >= 50:
        return 'Fair Match', 'score-fair'
    return 'Low Match', 'score-low'

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

def provider_links_for_skill(skill):
    q = quote_plus(skill)
    return [
        {'name': 'Coursera', 'url': f'https://www.coursera.org/search?query={q}'},
        {'name': 'Udemy', 'url': f'https://www.udemy.com/courses/search/?q={q}'},
    ]

def build_recommendations(skills):
    out = []
    for s in skills:
        out.append({'skill': s, 'providers': provider_links_for_skill(s)})
    return out

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
        job_skills = extract_skills(job_description, SKILL_KEYWORDS)
        missing_skills_list = []
        present_skills_list = []
        recommendations_list = []

        for text in resumes_texts:
            resume_skills = extract_skills(text, SKILL_KEYWORDS)
            missing = sorted(list(job_skills - resume_skills))
            present = sorted(list(job_skills & resume_skills))
            missing_skills_list.append(missing)
            present_skills_list.append(present)
            recommendations_list.append(build_recommendations(missing))

        percents = [int(round(float(s) * 100)) for s in scores]
        labels = []
        classes = []
        for p in percents:
            l, c = classify_match_label(p)
            labels.append(l)
            classes.append(c)

        results = pd.DataFrame({
            'Resume': resume_names,
            'Score': scores,
            'Percent': percents,
            'Label': labels,
            'ColorClass': classes,
            'Missing': missing_skills_list,
            'Present': present_skills_list,
            'Recommendations': recommendations_list
        })
        results = results.sort_values(by='Score', ascending=False)

        context = {
            'results': results.to_dict(orient='records'),
            'job_description': job_description
        }
    return render(request, 'index.html', context)
