<h1>Identifying Disaster-Related Tweets using Natural Language Processing and Core Machine Learning Algorithms</h1>

This repository contains the code and resources for a project focused on classifying tweets to determine whether they are related to a real-world disaster. In critical situations, social media platforms like Twitter become a vital source of real-time information. However, the sheer volume and noise (e.g., sarcasm, metaphors) make it challenging for humanitarian organizations to extract actionable intelligence.

This project implements and evaluates several core machine learning algorithms to automate the process of identifying disaster-related tweets, inspired by the research conducted by Aditi Mehta and Arshiya Garg.

**Reference Paper:** Mehta, A., Garg, A. (2024). Identifying Disaster-Related Tweets using Natural Language Processing and Core Machine Learning Algorithms. In: Sharma, H., et al. Sixth International Conference on Information and Communication Technology for Intelligent Systems. ICTIS 2024. Lecture Notes in Networks and Systems, vol 947. Springer, Cham. [https://doi.org/10.1007/978-3-031-91331-0_15](https://doi.org/10.1007/978-3-031-91331-0_15)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing and Cleaning](#1-data-preprocessing-and-cleaning)
  - [2. Feature Engineering (Vectorization)](#2-feature-engineering-vectorization)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Conclusion and Impact](#conclusion-and-impact)

---

## Project Overview

The primary objective of this project is to build a robust text classification model that can accurately distinguish between tweets announcing a real disaster and those that are not. By leveraging Natural Language Processing (NLP) techniques for text cleaning and feature extraction, we train and compare traditional machine learning models to find the most effective approach for this task. The successful implementation of such a system can significantly reduce manual effort and expedite the response time of emergency services.

## Dataset

The project utilizes the "Natural Language Processing with Disaster Tweets" dataset, originally sourced from a Kaggle competition. It contains over 10,000 hand-labeled tweets.

- **`train.csv`**: The raw training data.
- **`test.csv`**: The raw testing data for submission (target variable is not included).
- **`Cleaned_Data.csv`**: The output of our preprocessing pipeline, used for model training.

Each tweet in the dataset is labeled with a `target` variable:
- `1`: The tweet is about a real disaster.
- `0`: The tweet is not about a real disaster.

## Methodology

The project follows a standard machine learning pipeline, broken down into three main stages.

![Screenshot 2024-08-15 105518](https://github.com/user-attachments/assets/6fdd4655-b284-4b88-87db-8207fee9b4a8)

### 1. Data Preprocessing and Cleaning

Raw text data from tweets is inherently noisy. The `CleaningData.ipynb` notebook performs a series of essential preprocessing steps to standardize the text and prepare it for feature extraction:

- **Lowercasing:** All text is converted to lowercase.
- **Removal of URLs:** Web links are removed as they do not contribute to semantic meaning.
- **Removal of Punctuation and Special Characters:** Punctuation, numbers, and other non-alphabetic characters are stripped from the text.
- **Removal of Stopwords:** Common English words (e.g., "the", "a", "in") that provide little value for classification are removed using the NLTK library.
- **Lemmatization:** Words are converted to their base or dictionary form (e.g., "running" becomes "run") to consolidate different forms of the same word.

![Screenshot 2024-08-15 124645](https://github.com/user-attachments/assets/51cf57db-7517-402c-b587-43e78a3ec4db)

### 2. Feature Engineering (Vectorization)

Machine learning models cannot process raw text. Therefore, the cleaned text data must be converted into numerical feature vectors. This project explores two primary vectorization techniques:

- **Count Vectorizer (Bag-of-Words):** Creates a vocabulary of all unique words in the corpus and represents each tweet as a vector of word counts.
- **TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency):** An improvement over Count Vectorizer. It weighs word counts by how rarely they appear across the entire corpus, giving more importance to words that are significant for a specific tweet.

### 3. Model Training and Evaluation

Two different classification algorithms were trained and evaluated using the vectorized data:

- **Bernoulli Naive Bayes:** A probabilistic classifier based on Bayes' theorem, suitable for binary feature data (word presence/absence), which works well with the Bag-of-Words model.
- **Support Vector Machine (SVM):** A powerful classification algorithm that works by finding the optimal hyperplane that separates the two classes in a high-dimensional space. It is highly effective in text classification tasks, especially when paired with TF-IDF.

The performance of each model was measured using standard classification metrics, including **Accuracy** and the **F1-Score**, which provides a balanced measure of precision and recall.

## Results Summary

The project systematically evaluated four combinations of vectorizers and models. The performance metrics from the individual notebooks indicate that the combination of **TF-IDF Vectorization and the Support Vector Machine (SVM) classifier** achieved the best results.

![Screenshot 2024-08-16 092908](https://github.com/user-attachments/assets/13d18b73-8f1e-433a-8d1b-fb20b299116d)


The superior performance of the SVM with TF-IDF is attributed to TF-IDF's ability to capture word importance effectively and SVM's strength in handling high-dimensional, sparse data, which is characteristic of text.

## Repository Structure
```
Repository Structure
├── Cleaned_Data.csv
├── CleaningData.ipynb
├── Naive_Bayes_Bernoulli_Vectorizer.ipynb
├── Bernoulli Tf-Idf.ipynb
├── SVM Vectorizer.ipynb
├── SVM Tf-Idf.ipynb
├── README.md
└── train.csv
```



## How to Run

To reproduce the results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    Ensure you have Python 3 and Jupyter Notebook installed. The core libraries can be installed via pip.
    ```bash
    pip install pandas numpy scikit-learn nltk jupyter
    ```
    You will also need to download the NLTK data for stopwords and lemmatization. Run this in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

3.  **Run the notebooks:**
    - Start with `CleaningData.ipynb` to generate the `Cleaned_Data.csv` file.
    - You can then run any of the model notebooks (`SVM Tf-Idf.ipynb`, `Naive_Bayes_Bernoulli_Vectorizer.ipynb`, etc.) to train and evaluate the classifiers.

## Conclusion and Impact

This project successfully demonstrates that core machine learning algorithms, when combined with proper NLP preprocessing, can effectively automate the identification of disaster-related tweets. The findings confirm that the **Support Vector Machine model with TF-IDF features** provides a strong and reliable baseline for this classification task.

The real-world impact of such a system is significant. By automatically filtering and flagging relevant tweets, it can provide first responders and aid organizations with timely, critical information during emergencies, helping them to better understand a situation on the ground and allocate resources more efficiently.

Future work could involve exploring more advanced deep learning models, such as LSTMs or Transformers (e.g., BERT), which can capture more complex contextual relationships in language and potentially yield even higher accuracy.
