import streamlit as st
import re
import json
from pathlib import Path 

import nltk
nltk.download("punkt",quiet=True)
nltk.download("punkt_tab",quiet=True)
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="College FAQ Chatboat",page_icon="🌐",layout="centered")
st.title("COLLEGE FAQ CHATBOAT")
st.write("Welcome! Ask me your queries")

@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt_tab",quiet=True)
setup_nltk()  

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]","",text)
    tokens = word_tokenize(text)
    tokens = [token.strip()for token in tokens if token.strip()]
    return " ".join(tokens)

@st.cache_data
def load_faqs():
     file_path = Path(__file__).parent / "faq.json"
     with open (file_path,"r",encoding="utf-8")as file:
         faqs = json.load(file)
     if not faqs:  
         raise ValueError("The FAQ file is empty.")
     return faqs      

try:
    faqs = load_faqs()
except (FileNotFoundError, json.JSONDecodeError,ValueError) as error:
    st.error(f"Could not load FAQ data:{error}")
    st.stop()

questions = [faq["question"]for faq in faqs]    

clean_questions = [preprocess_text(questions)for questions in questions]

vectorizer = TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)

faq_vectors = vectorizer.fit_transform(clean_questions)

def get_answer(user_question):
    if not user_question.strip():
        return("Please enter a question.",0.0, None)

    cleaned_question = preprocess_text(user_question)
    if not cleaned_question:
        return("Please enter meaningful question.",0.0,None)

    user_vector = vectorizer.transform([cleaned_question])

    similarity_scores = cosine_similarity(user_vector,faq_vectors)[0]

    best_index = similarity_scores.argmax()
    best_score = float(similarity_scores[best_index])

    threshold = 0.20

    if best_score<threshold:
        return("Sorry, I couldn't find relevent answer." ,best_score,None)

    answer = faqs[best_index]["answer"]
    matched_question = questions[best_index]

    return answer , best_score , matched_question

# remove display previous messages part in this part 
if "messages" not in st.session_state:
    st.session_state.messages = []

user_question = st.chat_input("Enter your question here...")

if user_question:
    st.session_state.messages.append({"role": "user","content": user_question})

    with st.chat_message("user"):
        st.write(user_question)

    answer, score, matched_question = get_answer(user_question)

    with st.chat_message("assistant"):
        st.write(answer)

        if matched_question is not None:
            with st.expander("View Matched Question"):
                st.write(f"**Matched Question:** {matched_question}")
                st.write(f"**Similarity Score:** {score:.2f}")

    st.session_state.messages.append({"role": "assistant","content": answer})  

with st.sidebar:
    st.header("About")
    st.write("This is a College FAQ Chatboat that can answer your questions based on the provided FAQ data.")
    st.write("You can ask questions related to college policies, courses, admissions, and more.")
    st.write("The chatboat uses natural language processing techniques to find the most relevant answers from the FAQ data.")

    st.write(f"Total FAQ's:{len(faqs)}")  

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

