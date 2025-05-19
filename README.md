# Resume-Screening
Overview
The Automated Resume Screening App is a full-stack web application designed to streamline the hiring process by automatically analyzing resumes and matching them to job descriptions. Using basic natural language processing (NLP), the app extracts key skills and qualifications from uploaded resumes (PDF/text) and compares them to job requirements, providing a ranked list of candidates based on relevance.

This project demonstrates proficiency in full-stack development, API integration, file handling, and introductory NLP techniques, making it a valuable tool for recruiters and HR professionals.

Features
Resume Upload: Users can upload resumes in PDF or text format.

Job Description Input: Input job descriptions to define required skills and qualifications.

Skill Extraction: Automatically extracts skills and keywords from resumes using NLP (keyword matching or SpaCy).

Candidate Ranking: Matches resumes to job descriptions and ranks candidates by relevance score.

Responsive UI: Clean, mobile-friendly interface for ease of use.

Persistent Storage: Stores resumes and match results in a MongoDB database.

Export Results: Download match results as a CSV file for further analysis.

Tech Stack
Frontend: HTML<CSS<JS - Dynamic and responsive user interface.

Backend: Node.js, Express.js - RESTful API for processing resumes and job descriptions.

Database: MongoDB - Stores resume data and match results.

NLP: JavaScript-based keyword matching or SpaCy (via Python API integration) for skill extraction.

File Handling: Multer for handling file uploads, pdf-parse for extracting text from PDFs.

Styling: CSS with Bootstrap for a polished look.

