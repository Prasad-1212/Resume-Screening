from flask import Flask, render_template, request, jsonify
import pandas as pd
import pdfplumber
import docx
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Load dataset
csv_file = "candidates_extended_dataset.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, dtype=str)
else:
    df = pd.DataFrame(columns=["ID", "NAME", "SKILLS", "EXPERIENCE", "CERTIFICATION", "JOB_DESCRIPTION", "RELEVANT", "SENTIMENT_SCORE"])

df = df.astype(str).fillna("")

# Initialize NLP tools
vectorizer = TfidfVectorizer()
model = LogisticRegression()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Train Logistic Regression Model
if not df.empty and "SKILLS" in df.columns and "CERTIFICATION" in df.columns and "RELEVANT" in df.columns:
    X_train = vectorizer.fit_transform(df['SKILLS'] + " " + df['CERTIFICATION'])
    y_train = df['RELEVANT'].astype(int)
    if len(y_train) > 0:
        model.fit(X_train, y_train)

# Extract text from resumes
def extract_text_from_resume(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + " "
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + " "
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
    return text.strip()

# Extract features from text with sentiment analysis
def extract_features_with_sentiment(text, job_desc):
    name_pattern = r"(?i)(?:name[:\s]*)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    skills_pattern = r"(Python|Java|C\+\+|JavaScript|React|Node\.js|SQL|Machine Learning|AWS)"
    experience_pattern = r"(\d+)\s*years?"
    certification_pattern = r"(Certified in \w+)"

    name_match = re.findall(name_pattern, text)
    name = name_match[0] if name_match else "Unknown"
    skills = ", ".join(re.findall(skills_pattern, text)) if re.findall(skills_pattern, text) else "None"
    experience = re.findall(experience_pattern, text)
    experience = int(experience[0]) if experience else 0
    certifications = ", ".join(re.findall(certification_pattern, text)) if re.findall(certification_pattern, text) else "None"

    # Sentiment analysis on skills and experience context
    sentiment_score = analyze_sentiment(text, job_desc, skills)

    return name, skills, experience, certifications, sentiment_score

def analyze_sentiment(resume_text, job_desc, skills):
    if skills == "None":
        return 0.0
    
    # Tokenize and filter job description for key terms
    job_tokens = [word.lower() for word in word_tokenize(job_desc) if word.lower() not in stop_words]
    resume_tokens = word_tokenize(resume_text.lower())
    
    # Find context around skills in resume
    skill_words = skills.lower().split(", ")
    context_sentences = []
    for skill in skill_words:
        for i, token in enumerate(resume_tokens):
            if skill in token:
                start = max(0, i - 10)
                end = min(len(resume_tokens), i + 10)
                context_sentences.append(" ".join(resume_tokens[start:end]))
    
    if not context_sentences:
        return 0.0
    
    # Calculate sentiment for skill contexts
    total_sentiment = 0.0
    for sentence in context_sentences:
        scores = sia.polarity_scores(sentence)
        total_sentiment += scores['compound']  # Use compound score for overall sentiment
    
    avg_sentiment = total_sentiment / len(context_sentences)
    # Normalize to 0-100 scale (shift from -1 to 1 range to 0 to 1, then scale)
    normalized_sentiment = round(((avg_sentiment + 1) / 2) * 100, 2)
    return normalized_sentiment

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/speech")
def speech():
    return render_template("speechresume.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global df

    if "resumes" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("resumes")
    job_desc = request.form.get("job_desc", "").strip()
    
    if not job_desc:
        return jsonify({"error": "Job description is required"}), 400

    results = []

    for file in files:
        if file.filename == "":
            continue

        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        resume_text = extract_text_from_resume(file_path)
        if not resume_text:
            results.append({"filename": file.filename, "error": "Could not extract text", "score": 0, "sentiment_score": 0})
            continue

        name, skills, experience, certification, sentiment_score = extract_features_with_sentiment(resume_text, job_desc)

        if skills == "None" and certification == "None":
            relevance = 0
            similarity_score = 0
        else:
            if not df.empty and "SKILLS" in df.columns and "CERTIFICATION" in df.columns and "RELEVANT" in df.columns:
                resume_vector = vectorizer.transform([skills + " " + certification])
                relevance = model.predict(resume_vector)[0] if len(df) > 0 else 0
            else:
                relevance = 0

            if relevance == 1:
                try:
                    tfidf = TfidfVectorizer().fit_transform([resume_text, job_desc])
                    similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                    similarity_score = round(similarity_score * 100, 2)
                except Exception as e:
                    print(f"Error computing similarity: {e}")
                    similarity_score = 0
            else:
                similarity_score = 0

        try:
            last_id = df["ID"].max()
            if pd.isna(last_id) or not last_id.startswith("C"):
                unique_id = "C100"
            else:
                unique_id = f"C{int(last_id[1:]) + 1}"
        except:
            unique_id = "C100"

        new_entry = pd.DataFrame([{
            "ID": unique_id,
            "NAME": name,
            "SKILLS": skills,
            "EXPERIENCE": experience,
            "CERTIFICATION": certification,
            "JOB_DESCRIPTION": job_desc,
            "RELEVANT": int(relevance),
            "SENTIMENT_SCORE": sentiment_score
        }])

        if df is None or df.empty:
            df = new_entry
        else:
            df = pd.concat([df, new_entry], ignore_index=True)

        df.to_csv(csv_file, index=False)

        results.append({
            "filename": file.filename,
            "CID": unique_id,
            "name": name,
            "skills": skills,
            "experience": experience,
            "certification": certification,
            "relevance": int(relevance),
            "score": similarity_score,
            "sentiment_score": sentiment_score
        })

    results = sorted(results, key=lambda x: x.get("score", 0) + x.get("sentiment_score", 0), reverse=True)
    return jsonify({"resumes": results})

@app.route("/speech_intro_analyze", methods=["POST"])
def speech_intro_analyze():
    data = request.get_json()
    job_desc = data.get("job_desc", "").strip()
    intro_text = data.get("intro_text", "").strip()

    if not job_desc or not intro_text:
        return jsonify({"error": "Job description and introduction text are required"}), 400

    name, skills, experience, certification, sentiment_score = extract_features_with_sentiment(intro_text, job_desc)

    if skills == "None" and certification == "None":
        relevance = 0
        similarity_score = 0
    else:
        if not df.empty and "SKILLS" in df.columns and "CERTIFICATION" in df.columns and "RELEVANT" in df.columns:
            intro_vector = vectorizer.transform([skills + " " + certification])
            relevance = model.predict(intro_vector)[0] if len(df) > 0 else 0
        else:
            relevance = 0

        if relevance == 1:
            try:
                tfidf = TfidfVectorizer().fit_transform([intro_text, job_desc])
                similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                similarity_score = round(similarity_score * 100, 2)
            except Exception as e:
                print(f"Error computing similarity: {e}")
                similarity_score = 0
        else:
            similarity_score = 0

    return jsonify({
        "intro_score": similarity_score,
        "sentiment_score": sentiment_score
    })

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)