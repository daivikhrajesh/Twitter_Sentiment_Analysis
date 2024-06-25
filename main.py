import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Function to load and preprocess the dataset
def load_and_preprocess(filename):
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(filename, names=col_names, encoding='latin-1')
    
    print("Dataset Shape:", df.shape)
    print(df.head(5))
    print('\nMissing Values:\n', df.isnull().sum())
    print('\nTarget Value Counts:\n', df['target'].value_counts())
    
    # Replace target values (4 -> 1)
    df['target'].replace({4: 1}, inplace=True)
    print('\nUpdated Target Value Counts:\n', df['target'].value_counts())
    
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    return df

# Define a function for preprocessing

def preprocess_text(content):
    port_stem = PorterStemmer()
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  
    stemmed_content = stemmed_content.lower()  
    stemmed_content = stemmed_content.split()  

    # Stemming and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in set(stopwords.words('english'))]  
    return ' '.join(stemmed_content) 


if __name__ == '__main__':
    # Load and preprocess the dataset
    filename = 'training.1600000.processed.noemoticon.csv'
    df = load_and_preprocess(filename)
    
    # Split data into training and testing sets
    X = df['clean_text'].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    
    # Convert textual data into numeric using TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions and accuracy scores
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('Training Accuracy:', train_accuracy)
    print('Testing Accuracy:', test_accuracy)
