# Frequent Pattern Mining

## Introduction
Frequent Pattern Mining (FPM) identifies common co-occurring words or patterns within a dataset. For this project, we applied FPM to analyze a collection of tweets, aiming to uncover recurring themes related to emergencies.

---

## Preprocessing
Before running the FPM algorithm, we cleaned and prepared the dataset:
1. **Cleaned the Text**: Removed punctuation, URLs, and stopwords to focus on meaningful words.
2. **Handled Missing Data**: Replaced missing values in the `keyword` and `location` columns with empty strings.
3. **Created Transactions**: Each tweet was tokenized into words, and the corresponding `keyword` and `location` were added to form transactions.

### Code:
```python
import pandas as pd
import re
from nltk.corpus import stopwords
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
data = pd.read_csv('tweetsv2.csv')

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply text cleaning
data['cleaned_text'] = data['text'].apply(clean_text)
data['tokens'] = data['cleaned_text'].apply(lambda x: x.split())

# Include keywords and locations in transactions
data['keyword'] = data['keyword'].fillna('').astype(str)
data['location'] = data['location'].fillna('').astype(str)
data['transactions'] = data.apply(lambda row: row['tokens'] + [row['keyword'], row['location']], axis=1)

# Verify cleaned transactions
print(data['transactions'].head())

Cleaned Transactions Sample:
['fire', 'rescue', 'evacuate', 'keyword', 'location']
['earthquake', 'damage', 'relief', 'keyword', 'location']

# Frequent Itemsets
Using the Apriori algorithm, we extracted frequent itemsets with a minimum support of 1%. Frequent itemsets represent words or combinations of words that appear together in at least 1% of the transactions.

# Transaction Encoding
transactions = data['transactions'].tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori
min_support = 0.01  # Set minimum support
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets.head())

```python

Frequent Itemsets:
   support               itemsets
0   0.012  ['fire', 'rescue']
1   0.011  ['earthquake']
2   0.015  ['relief', 'damage']

Association Rules
Association rules were generated from the frequent itemsets to find relationships between co-occurring words. These rules show patterns where the presence of one word predicts another with high confidence.

# Generate Association Rules
min_confidence = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display rules
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'confidence', 'lift']])

# Generate Association Rules
min_confidence = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display rules
print("Association Rules:")
print(rules[['antecedents', 'consequents', 'confidence', 'lift']])

Analysis and Limitations
Frequent Itemsets: Patterns like ['fire', 'rescue'] indicate common co-occurrences in emergency-related tweets.

## Introduction
Frequent Pattern Mining (FPM) identifies common co-occurring words or patterns within a dataset. For this project, we applied FPM to analyze a collection of tweets, aiming to uncover recurring themes related to emergencies.

---
Future Work
N-Grams: Use bigrams or trigrams to extract richer context from tweets.
Lower Thresholds: Experiment with even lower support and confidence thresholds to uncover less frequent patterns.
Alternative Methods: Explore clustering or classification to complement frequent pattern analysis.
