import string
import re
import pickle 
from tensorflow import keras 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import streamlit as st


def check(query):
    filtered_text = preprocess_text(query)

    with open('asset/tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text = tokenizer.texts_to_sequences([filtered_text])
    max_sequence_length = 50

    padded_sequences = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
    
    model = keras.models.load_model('asset/model.h5')
    
    probability = model.predict(padded_sequences)[0]
    
    if probability < 0.5:
        return 'negative'
    return 'positive'

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between']


def preprocess_text(text):
    #Remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove non-alphanumeric characters and special characters
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    
    # Remove user mentions and hashtags
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    
    # Remove punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # processed_text = applying_stemming(filtered_tokens)
    
    return filtered_tokens


st.title("Sentiment Analysis")


user_input = st.text_input(
    'Enter the text whose Sentiment you want to analyze',
)

if 'last_text_input' not in st.session_state: 
    st.session_state.last_text_input = ''
    
if st.button("Analyze") or (user_input and st.session_state.last_text_input != user_input): 
    result = check(user_input)
    st.write(" The model predicts that the statement is ", result)