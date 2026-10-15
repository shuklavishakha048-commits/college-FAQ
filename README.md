College FAQ Chatbot

Project Description

A simple FAQ chatbot built using Python, NLP, and Streamlit. It matches user questions with predefined FAQs and provides the most relevant answer using TF-IDF and cosine similarity.

Features

- Text preprocessing using NLTK
- FAQ data stored in JSON format
- Question matching using TF-IDF and cosine similarity
- Fallback response for unmatched questions
- Interactive chat interface using Streamlit
- Chat history and Clear Chat option
- Similarity score and matched FAQ display

Technologies Used

- Python
- Streamlit
- NLTK
- Scikit-learn
- JSON

Project Structure

CodeAlpha_FAQ_Chatbot/
├── chatboat.py
├── faqs.json
└── README.md

Installation

Install the required libraries:

pip install streamlit nltk scikit-learn

How to Run

Run the following command in the terminal:

python -m streamlit run chatboat.py

Open the local URL displayed in the terminal to use the chatbot.

How It Works

1. Loads predefined questions and answers from "faqs.json".
2. Preprocesses text using NLTK.
3. Converts FAQ questions into numerical vectors using TF-IDF.
4. Calculates cosine similarity between the user's question and stored FAQs.
5. Displays the most relevant answer if the similarity score meets the threshold.

Future Improvements

- Add more FAQs.
- Support multiple languages.
- Improve question matching.
- Add voice input and text-to-speech.