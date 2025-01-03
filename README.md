# Sentiment Analysis Repository

This repository contains scripts for performing sentiment analysis on text data using various machine learning and deep learning techniques. Each script demonstrates a unique approach to preprocessing and modeling.

## File Descriptions

### 1. `sentiment_analysis_lstm.py`
This script performs sentiment analysis using LSTM (Long Short-Term Memory) neural networks and word embeddings. Key features include:

- **Preprocessing**: Removes HTML tags, punctuations, and stopwords. Applies lemmatization.
- **Embedding Layer**: Utilizes Keras's `Embedding` layer for vector representation.
- **Model Architecture**: Bidirectional LSTM with dense layers.
- **Dataset**: IMDB dataset for sentiment analysis.

#### Steps:
1. Load and preprocess the IMDB dataset.
2. Tokenize and pad sequences for input to the model.
3. Define and train a Bidirectional LSTM model.
4. Evaluate the model on a test set and print metrics.

### 2. `word2vec_sentiment.py`
This script combines Word2Vec embeddings with a deep learning model for sentiment analysis. Key features include:

- **Custom Embeddings**: Trains Word2Vec embeddings using the Gensim library.
- **Embedding Initialization**: Uses pretrained Word2Vec embeddings in the Keras embedding layer.
- **Model Architecture**: Bidirectional LSTM with dense layers.
- **Dataset**: IMDB dataset for sentiment analysis.

#### Steps:
1. Train Word2Vec on preprocessed corpus.
2. Prepare an embedding matrix for initializing the embedding layer.
3. Build and train a Bidirectional LSTM model.
4. Evaluate performance on a test set.

### 3. `naive_bayes_sentiment.py`
This script applies traditional machine learning techniques for sentiment analysis. Key features include:

- **Preprocessing**: Removes HTML tags, punctuations, and stopwords. Applies lemmatization.
- **Feature Extraction**: Utilizes TF-IDF for converting text into numerical features.
- **Classifier**: Multinomial Naive Bayes for sentiment prediction.
- **Dataset**: IMDB dataset for sentiment analysis.

#### Steps:
1. Preprocess the text data.
2. Extract features using TF-IDF vectorization.
3. Train a Multinomial Naive Bayes model.
4. Evaluate the model using metrics like accuracy and confusion matrix.

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - nltk
  - sklearn
  - tensorflow
  - gensim

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run any script:
   ```bash
   python <script_name.py>
   ```

Replace `<script_name.py>` with `sentiment_analysis_lstm.py`, `word2vec_sentiment.py`, or `naive_bayes_sentiment.py` depending on the desired approach.

## Dataset
The scripts use the IMDB dataset for sentiment analysis. Ensure the dataset file `imdb_dataset.csv` is present in the same directory as the scripts.

## Results
- **Deep Learning Models**: Achieve high accuracy using LSTM and Word2Vec embeddings.
- **Naive Bayes Model**: Provides a simpler and faster solution with decent performance.

## License
This repository is licensed under the MIT License.

