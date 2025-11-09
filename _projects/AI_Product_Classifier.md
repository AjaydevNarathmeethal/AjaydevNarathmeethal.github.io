---
layout: page
title: Hybrid AI for Retail Product Categorization
description: A hybrid LLM + rule-based system for large-scale retail product classification, focused on the Food sector.
img: assets/img/proj_AI_prod_classif/1.jpg
importance: 1
category: work
related_publications: false
---

📂 [GitHub Repo Link](https://github.com/AjaydevNarathmeethal/Hybrid-AI-for-Retail-Product-Categorization)

## 1. Introduction
The classification of retail product data, particularly in the food sector, presents unique challenges due to the ambiguity of product names, inconsistent labeling, and absence of brand cues. In this project, we tackled the problem of categorizing over 2 million Amazon product entries into Food and Non-Food categories, and further into subcategories, using a hybrid approach combining LLMs and rule-based refinement.
We began by leveraging Amazon’s intuitive and expansive category structure as the foundation for our taxonomy. To reduce labeling costs and scale efficiently, we adopted a DeepSeek-inspired strategy—using the open-source LLaMA3 model locally with Ollama, which enabled GPU-based inference without relying on commercial APIs.
To generalize the inference across all remaining data, we fine-tuned DeBERTa, a context-aware transformer, trained on over 249,000 labeled samples. The model achieved over 96% precision in Level 1 classification (Food vs. Non-Food), with substantial improvements observed in Level 2 and 3 subcategories.
This project demonstrates how lightweight infrastructure, human-in-the-loop rule refinement, and open-source LLMs can be orchestrated to deliver production-grade classification at scale. We further built a Streamlit-based web demo to allow real-time inspection of classification outputs, subcategory breakdowns, and model decisions—bridging technical results with practical usability for stakeholders.



## 2. Data Sources and Key Variables
To build a reliable product classification system, we collected product data from multiple major U.S. retailers:
- **Amazon**: Chosen for its intuitive and detailed category hierarchy, which served as the primary reference taxonomy.
- **Target & Walmart**: Supplemented the dataset to increase product diversity and improve generalizability across different e-commerce platforms.

From these sources, we extracted:
- Product Name
- Search Terms

Using these inputs, we designed a multi-level taxonomy:
- Level 1: Food vs. Non-Food
- Level 2: 8 key food subcategories (e.g., Beverages, Snacks)

This multi-source approach helped ensure the model could generalize beyond Amazon and adapt to broader retail contexts.


## 3. Data Preprocessing
We began by importing raw shopper log data from multiple files and merged it with panelist demographic data. After removing missing values and duplicates, the dataset was chronologically sorted by panelist and session.
We then segmented sessions using time thresholds—30 minutes for searches and 60 minutes for other events—to identify distinct shopping behaviors. Using cosine similarity and a time-decay function, we linked search terms to related product views, while unmatched records were labeled as either "direct_access" or "search_only."
Finally, the events were grouped and restructured into session-level units, and we validated the cleaned dataset by checking for missing values, abnormal text lengths, and data type consistency.


## 4. Food Categorization Level1 Labeling
This Python script analyzes shopper behavior data by loading search terms and product names from clickstream logs. It focuses on identifying whether a product is related to Food or Non-Food using simple keyword-matching rules (e.g., "juice", "cookie", "shampoo", etc.).

The script:
- Cleans and deduplicates relevant columns.
- Applies a rule-based classification function (classify_food_level1) that scans for predefined keywords.
- Labels each item at Level 1 with "Food", "Non-Food", or "Unknown".
- Saves the final results to a CSV file for downstream categorization or analysis.

This serves as a lightweight but effective first-pass filter before applying deeper hierarchical classification (Level 2 and 3). 

```python

try:
    classification_df = df[['Panelist id', 'Retailer property name', 'Session',
                             'Event type', 'Search term', 'Product name']].copy()
except KeyError as e:
    raise Exception(f"Missing column: {e}")

classification_df['Search term'] = classification_df['Search term'].fillna('Direct access')
classification_df['Product name'] = classification_df['Product name'].fillna('')
classification_df = classification_df.drop_duplicates(subset=['Search term', 'Product name'])

# 3. Level 1 classification function (rule-based)

def classify_food_level1(search_term, product_name):
    text = f"{search_term} {product_name}".lower()
    food_keywords = ['Juice', 'Cookie', 'Snack', 'Cereal', 'Coffee', 'Chocolate', 'Soda', 'Milk', 'Candy', 'Chips', 'Crackers']
    non_food_keywords = ['Shampoo', 'Detergent', 'Toilet', 'face mask', 'Lotion', 'Batteries', 'Napkin', 'Toothpaste', 'Cleaner']
    
    if any(word in text for word in food_keywords):
        return 'Food'
    elif any(word in text for word in non_food_keywords):
        return 'Non food'
    else:
        return 'Unknown'

# 4. Apply classification

print("Level 1 Sorting...")
classification_df['Level1 category'] = classification_df.apply(
    lambda row: classify_food_level1(row['Search term'], row['Product name']), axis=1)
	
```

## 5. Food Categorization Level1 Deberta Learning- Inference
This project builds a hierarchical classification system to categorize shopper log data into structured food-related categories. It combines rule-based preprocessing with a fine-tuned DeBERTa-v3 language model.

### 1. Data Preprocessing
- Shopper logs are cleaned and deduplicated, focusing on product_name and search_term.
- Input text is structured to emphasize product names while preserving user search context.
- Categories are normalized and encoded using a predefined taxonomy:
	- Level 1: Food / Non-Food
	- Level 2 (only for Food): Beverages, Snacks, Breakfast, Pantry Staples, etc.
- Code in 5_1.deberta_preprocess.py

### 2. Model Training
- A DeBERTa-v3-base model is trained for hierarchical classification using multi-task learning.
- The model uses separate classification heads for Level 1 and Level 2 outputs.
- Mixed precision (FP16), gradient accumulation, and warmup scheduling are used for optimized training on RTX 3070 GPUs.
- Code in 5_2.deberta_learning.py

### 3. Inference Pipeline
- The model classifies new, unlabeled shopper entries.
- If a product is predicted as Food (Level 1), it is further classified into Level 2 subcategories.
- Sample outputs include summary statistics and are saved for integration with downstream applications.
- Code in 5_3.deberta_inference.py

## 6. Food Categorization Level2 Llama3 
This project implements an automated classification system to assign Level 2 food categories based on user search terms and product names using an LLM (LLaMA3) through the Ollama API. The system is designed for efficient batch processing with GPU-aware optimization.

### 1. Objective
To classify product-level shopper data into structured Level 2 food categories using LLM reasoning, specifically for entries where Level 1 is already labeled as "Food".

### 2. Key Features
- LLM Integration: Utilizes llama3:latest model via Ollama API to infer Level 2 categories from product and search term pairs.
- Batch Inference with Dynamic Adjustment: Adaptive batch sizing based on real-time GPU, CPU, and memory usage.
- Caching: Avoids redundant inference for repeated product entries using key-based caching.
- Prompt Engineering: Includes an optimized few-shot prompt with classification instructions and edge-case guidelines.
- Resumable Processing: Supports automatic checkpoint loading to resume from previous sessions.
- Async Checkpoint Saving: Saves interim results periodically using multi-threading to avoid data loss.
- Categorical Validation: Normalizes and validates outputs against a predefined taxonomy to ensure integrity.

### 3. System Requirements
- Python with packages: pandas, torch, requests, tqdm, psutil
- Running Ollama API server locally (port 11434)
- GPU recommended (RTX 3070 or higher); dynamically adjusts batch size based on GPU memory (default: 3)

### 4. Output
- Final CSV file containing search_term, product_name, level1_category, and predicted level2_category
- Intermediate checkpoint files for fault tolerance
- Code in 6_level2_reclassificationv3.py

## 7. Food Categorization Interface with Streamlit (DeBERTa + LLaMA3 Hybrid)
### Objective
To provide an interactive browser-based interface for food classification using a hybrid model approach.

### Summary
The Streamlit app enables users to input product descriptions and classify them as Food or Non-Food (Level 1) using a fine-tuned DeBERTaV2 model. If the product is food, the app supports two methods for Level 2 classification:
- DeBERTa: Local model-based prediction
- LLaMA3 (Ollama API): Prompt-based categorization using few-shot examples and category guidelines

### Features
- Real-time text input and prediction interface
- Dual model selection for Level 2: DeBERTa or LLaMA3
- Debug mode for API tracing
- Result summary with classification history logging
- Code in 7_ Food Categorizer Streamlit App.py


## 8. Key Lessons & Future Directions:

Over the course of this project, we learned to implement a full stack text processing model, starting from storing the data to cleaning to categorizing using a LLM + rule-based hybrid model to implementing the result in a web page.  We also tasted the strength and limitations of current LLMs when it comes to unsupervised categorizing a raw text data.  We understood that a hybrid approach incorporating a few hard coded rules are most efficient in providing the best results. 

For the future improvements of the model, the accuracy of level1 categorization can be pushed to 96% by refining the prompt used in LLM.  Further an independent BERT-based model can be built to improve the accuracy and performance of the overall categorization of this model.  Also, using a cloud system with higher processing power, this model can be scaled to perform the categorization of larger datasets with improved speed and accuracy.


## Demo of Classifier deployed in Stremlit

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_AI_prod_classif/Recalling Food Classification Levels.gif" title="Demo of classifier" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Demo of classifier
</div>
