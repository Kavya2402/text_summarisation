import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



import PyPDF2


def extract_pdf(path):
    """
    This function extracts the text from the uploaded PDF file.
    
    Parameters
    ----------
    path : str
        The path to the uploaded PDF file.
    """
    pdf_file = open(path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    extracted_text = ''

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()

    pdf_file.close()

    return extracted_text



def text_summarization(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Preprocess the sentences
    stopwords_list = set(stopwords.words("english"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in stopwords_list]
        preprocessed_sentences.append(" ".join(words))
    
    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    # Calculate sentence scores based on TF-IDF
    sentence_scores = {}
    for i in range(len(sentences)):
        sentence_scores[i] = tfidf_matrix[i].sum()
    
    # Sort the sentences by score in descending order
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # Select the top N sentences for the summary
    summary_sentences = sorted_sentences[:3]  # Change the number to adjust the length of the summary
    
    # Generate the summary
    summary = ""
    for sentence_index in summary_sentences:
        summary += sentences[sentence_index] + " "
    
    return summary


text = extract_pdf('text_summarisation/text_file.pdf')
summary = text_summarization(text)
print(summary)
