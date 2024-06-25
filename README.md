# Sentiment Analysis with Logistic Regression

This project demonstrates sentiment analysis using Logistic Regression on a dataset of tweets. The goal is to predict whether a tweet has a positive sentiment (1) or a negative sentiment (0).

## Dataset

The dataset used for this project is from Twitter and contains 1.6 million tweets. It was processed to remove emoticons and contains columns such as 'target' (sentiment label), 'text' (tweet content), and 'clean_text' (preprocessed tweet text).
 
### Dataset Details

- **Shape:** [Number of rows, Number of columns]
- **Missing Values:** [Details on any missing values]
- **Target Value Counts:** [Counts of each sentiment class]

## Preprocessing

Text preprocessing involves:
- Converting to lowercase
- Removing non-alphabetic characters
- Tokenization and stemming
- Removing stopwords

## Model Training

The textual data is transformed into numeric using TF-IDF vectorization. A Logistic Regression model is trained on the vectorized data.

## Usage

To run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords:
   ```
   python -m nltk.downloader stopwords
   ```

4. Download the dataset (`training.1600000.processed.noemoticon.csv`) and place it in the project directory.
    https://www.kaggle.com/datasets/kazanova/sentiment140/data

5. Run the script:
   ```
   python main.py
   ```

6. View results:
   - Training Accuracy
   - Testing Accuracy


## Dependencies

- pandas
- numpy
- nltk
- scikit-learn

## Author

Daivikh Rajesh Mysuru
