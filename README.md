# Music Genre Classification based on Lyrics
### Combined Model based on statistic structure, semantic structure and lyrics similarity encoding

### 1. Statistic Structure

***Extracting statistical features based on TF-IDF for different n-grams at word and character level***

### 2. Semantic Structure

***To be able able to extract and quantify various semantinc relations from lyrics, we use SOTA transformer models***

### 3. Lyrics Similarity 

***For lyrics similarity, we use a pretrained transformer model for extracting similarity embeddings between lyrics***

# Final Solution

Voting system between various Machine Learning algorithms and Transformer Models over different inputs.

# Models 

### Machine Learning models
- Light Gradient Boosting
- Support Vector Machines
- Logistic Regression
- Random Forest

### Transformer models
- RoBERTa
- XLNet

### Similarity network
- [stsb-roberta-large](https://huggingface.co/sentence-transformers/stsb-roberta-large) 

# Ensembling Methods
- [x] **Voting system (Final Solution)**
- [x] Weighted voting system
- [x] Stacking
