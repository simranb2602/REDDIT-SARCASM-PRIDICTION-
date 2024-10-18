import os
import pickle
import re
import string
import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer 
from flask import Flask, render_template, request, url_for, jsonify
from scipy.sparse import hstack, csr_matrix
from CustomLSTMClassifier import CustomLSTMClassifier

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")
 
# Initialize the NLTK tokenizer, lemmatizer, and stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Import stopwords and define custom set
stops = set(stopwords.words('english')) - {'no', 'not', 'nor', 'against', 'above', 'below', 'off', 'own'}
 
# Function to preprocess text
def preprocess_text(text):

    def clean_text(text):

        # Cleaning the text by removing links, usernames, HTML tags, expansion of words, username removal, etc.
        text = str(text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Remove HTML tags
        text = re.sub("<.*?>", " ", text)

        # Remove numbers
        text = re.sub(r"[0-9]+", " ", text)

        # Remove reddit handles
        text = re.sub(r"@[A-Za-z0-9]+", " ", text)

        # Replace contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)

        # Remove special characters
        text = text.replace('\\r', ' ')
        text = text.replace('\\"', ' ')
        text = text.replace('\\n', ' ')

        return text
    
    #function to lowercase all except all caps word
    def lowercase_except_caps(sentence):
        #print(sentence)
        words = sentence.split()
        modified_words = []
        for word in words:
            if word.isupper():
                modified_words.append(word)
            else:
                modified_words.append(word.lower())
        return " ".join(modified_words)
    
    # Function to expand abbreviations
    def expand_abbr(text, abbr_dict):
        pattern1 = r"\b([A-Z]{2,}(?!\.[A-Z]{2,}))\b"
        pattern2 = r"\b([A-Z]{2,}\.?)\b"
        combined_pattern = re.compile("|".join([pattern1, pattern2]))

        detected_abbr = []

        def expand_match(match):
            matched_abbr = match.group()
            detected_abbr.append(matched_abbr)
            for entry in abbr_dict:
                if matched_abbr.upper() == entry['Abbr']:
                    return entry['Meaning']
            return matched_abbr

        expanded_text = combined_pattern.sub(expand_match, text)
        return expanded_text, detected_abbr
    
    # Function to replace slangs
    def replace_slangs(sentence, slangs_dict):
        words = sentence.split()
        replaced_words = []
        detected_slangs = []
        for word in words:
            meaning = slangs_dict.get(word, word)
            if meaning != word:
                detected_slangs.append((word, meaning))
            replaced_words.append(meaning)
        return ' '.join(replaced_words), detected_slangs

    # Function to check if a string contains emojis
    def has_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return bool(emoji_pattern.search(text))

    # Apply stopword removal to the emoji_replaced_comment column
    def remove_stopwords(comment):
        return [word for word in ast.literal_eval(comment) if word.lower() not in stops]

    # Function to replace emojis with their meanings
    def replace_emojis(text):
        return emoji.demojize(text)

    #Clean input text
    text = clean_text(text)
    print('Clean input text: ' ,text)

    #Lowercase text to all except all caps word
    text = lowercase_except_caps(text)
    print('Lowercase text: ',text)

    # Load abbreviation and slang dictionaries and stopwords
    abbrslang = pd.read_csv('Abbr.csv')
    abbrslangDict = abbrslang.to_dict(orient='records')
    
    # Expand abbreviations
    text, _ = expand_abbr(text, abbrslangDict)
    print('Expand abbreviations: ',text)

    slangs_df = pd.read_csv('Slangs.csv')
    slangs_dict = dict(zip(slangs_df['slangs'], slangs_df['meanings']))
    # Replace slangs
    text, _ = replace_slangs(text, slangs_dict)
    print('Replace slangs: ',text)

    # Remove punctuation marks
    text = ''.join([char for char in text if char not in string.punctuation or char in ['?', '!', '...']])
    print('Remove punctuation marks: ',text)

    # Check for emojis and replace them with their meanings
    if has_emoji(text):
        text = replace_emojis(text)
    print('Check for emojis: ',text)

    # Tokenize the text
    tokens = word_tokenize(text)
    print('Tokenized text: ',tokens)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stops]
    print('stopwords Removed: ',tokens)

    return tokens

 
# Function to compute average GloVe embeddings for a text
def compute_average_glove_embedding(text_list, nlp_model):
    avg_embeddings = []
    for text in text_list:
        doc = nlp_model(str(text))
        # Get vectors for each token in the text
        word_vectors = [token.vector for token in doc if not token.is_stop]
        if word_vectors:
            # Compute average embedding
            average_embedding = np.mean(word_vectors, axis=0)
            avg_embeddings.append(average_embedding)
        else:
            # Append zeros if no valid word vectors found
            avg_embeddings.append(np.zeros(nlp_model.vocab.vectors_length))
    return np.array(avg_embeddings)

# Function to lemmatize input text
def lemma(comment):
    lemmatized_comments = []
    for sentence_tokens in comment:
        # Join the list of tokens into a single string
        text = ' '.join(sentence_tokens)
        # Tokenize the text
        words = word_tokenize(text)
        # Lemmatize and remove stopwords
        cleaned_words = [token.lemma_ for token in nlp(' '.join(words)) if token.text not in stops]
        # Join the cleaned words back into a single string
        cleaned_text = ' '.join(cleaned_words)
        lemmatized_comments.append(cleaned_text)
    return np.array(lemmatized_comments)

# Function to create TF-IDF vectorizer for lemmatized input text
def vectorize(comment):
    vectorizer_comment = TfidfVectorizer()
    TfIdfMatrix_comment = vectorizer_comment.fit_transform(comment)
    return TfIdfMatrix_comment.toarray()
     
def predict_sarcasm(input_data):

    print("Number of input sentences:", len(input_data))
    preprocessed_data = [preprocess_text(sentence) for sentence in input_data]
    glove_embedding = compute_average_glove_embedding(preprocessed_data, nlp)

    print("Shape of glove_embedding matrix:", glove_embedding.shape)
    lemmatized_input = lemma(preprocessed_data)
    vectorized_data = vectorize(lemmatized_input)
    print("Shape of vectorized_data matrix:", vectorized_data.shape)

    # Convert the GloVe embeddings to CSR matrices for lemma_comment
    glove_embeddings_matrix = csr_matrix(np.vstack(glove_embedding))
    combined_matrix = hstack([vectorized_data, glove_embeddings_matrix])
    print("Shape of combined matrix:", combined_matrix.shape)

    # Pad the combined_matrix to match the expected number of features
    model = pickle.load(open('fore.pkl', 'rb'))
    num_expected_features = len(model.estimators_[0].feature_importances_)  # Number of features used by one estimator
    num_combined_features = combined_matrix.shape[1]

    if num_combined_features < num_expected_features:
        # Create a matrix of zeros to pad the combined_matrix
        padding = np.zeros((combined_matrix.shape[0], num_expected_features - num_combined_features))
        combined_matrix_padded = hstack([combined_matrix, csr_matrix(padding)])
    else:
        combined_matrix_padded = combined_matrix

    predicted_sarcasm = model.predict(combined_matrix_padded)
    return predicted_sarcasm

 
app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('fore.pkl', 'rb')) 
@app.route('/')

def first():
    return render_template("first.html")
 
@app.route('/index')

def index():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])

def predict():
    input_sentence = request.form['input_sentence']
    input_data = [input_sentence]
    
    try:
        # Predict sarcasm
        predicted_sarcasm = predict_sarcasm(input_data)
        if predicted_sarcasm == 1:
            predicted_sarcasm = "Comment is sarcastic"
        else:
            predicted_sarcasm = "Comment is non-sarcastic"
        return render_template('index.html', prediction=predicted_sarcasm)
    except Exception as e:
        # Handle any exceptions and return an error message
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', prediction=error_message)
     
if __name__ == '__main__':

    app.run(debug=True)
