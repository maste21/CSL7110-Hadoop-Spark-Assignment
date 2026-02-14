#!/usr/bin/env python3
"""
CSL7110 Assignment - Question 11 (FIXED)
TF-IDF and Book Similarity - Corrected for SparseVector handling
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import SparseVector, DenseVector, Vectors
import numpy as np
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("TFIDF_BookSimilarity") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("="*60)
print("QUESTION 11: TF-IDF and Book Similarity")
print("="*60)

# ------------------------------------------------------------
# STEP 1: Load the Gutenberg dataset
# ------------------------------------------------------------
print("\n📂 STEP 1: Loading Gutenberg dataset...")

def load_books(directory_path):
    """Load all .txt files from directory"""
    from pyspark.sql import Row
    
    books = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:50000]  # First 50k chars
                books.append(Row(
                    file_name=filename,
                    text=content
                ))
                print(f"  ✓ Loaded: {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
    
    return spark.createDataFrame(books)

books_df = load_books("gutenberg_dataset")
print(f"\n✅ Loaded {books_df.count()} books")
books_df.select("file_name").show(5, truncate=False)

# ------------------------------------------------------------
# STEP 2: Preprocessing - Remove Gutenberg header/footer
# ------------------------------------------------------------
print("\n🔧 STEP 2: Removing Gutenberg header/footer...")

def extract_main_content(text):
    """Extract content between START and END markers"""
    # Find start of content
    start_markers = ["*** START OF", "***START OF", "*END*THE SMALL PRINT"]
    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            start_idx = pos
            break
    
    # Find end of content
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]
    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break
    
    return text[start_idx:end_idx]

# Register UDF
extract_udf = udf(extract_main_content, "string")
content_df = books_df.withColumn("content", extract_udf(col("text")))

# ------------------------------------------------------------
# STEP 3: Clean text - lowercase, remove punctuation
# ------------------------------------------------------------
print("\n🧹 STEP 3: Cleaning text (lowercase, remove punctuation)...")

cleaned_df = content_df.select(
    col("file_name"),
    # Convert to lowercase
    lower(col("content")).alias("text_lower")
).select(
    col("file_name"),
    # Remove punctuation (keep only letters and spaces)
    regexp_replace(col("text_lower"), "[^a-zA-Z\\s]", " ").alias("text_no_punct")
).select(
    col("file_name"),
    # Remove extra spaces
    regexp_replace(col("text_no_punct"), "\\s+", " ").alias("clean_text")
)

print("✅ Text cleaning complete")
cleaned_df.select("file_name", "clean_text").show(3, truncate=60)

# ------------------------------------------------------------
# STEP 4: Tokenization
# ------------------------------------------------------------
print("\n🔪 STEP 4: Tokenizing text into words...")

tokenizer = RegexTokenizer() \
    .setInputCol("clean_text") \
    .setOutputCol("tokens") \
    .setPattern("\\s+") \
    .setMinTokenLength(2)

tokenized_df = tokenizer.transform(cleaned_df)
print("Sample tokens:")
tokenized_df.select("file_name", "tokens").show(3, truncate=40)

# ------------------------------------------------------------
# STEP 5: Remove stop words
# ------------------------------------------------------------
print("\n🚫 STEP 5: Removing stop words...")

remover = StopWordsRemover() \
    .setInputCol("tokens") \
    .setOutputCol("filtered_tokens") \
    .setCaseSensitive(False)

filtered_df = remover.transform(tokenized_df)
print("After stop word removal:")
filtered_df.select("file_name", "filtered_tokens").show(3, truncate=40)

# ------------------------------------------------------------
# STEP 6: TF-IDF Calculation
# ------------------------------------------------------------
print("\n📊 STEP 6: Calculating TF-IDF...")

# 6.1 Term Frequency (TF)
hashing_tf = HashingTF() \
    .setInputCol("filtered_tokens") \
    .setOutputCol("tf_features") \
    .setNumFeatures(65536)

tf_df = hashing_tf.transform(filtered_df)

# 6.2 Inverse Document Frequency (IDF)
idf = IDF() \
    .setInputCol("tf_features") \
    .setOutputCol("tfidf_features") \
    .setMinDocFreq(2)

idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# 6.3 Normalize vectors
normalizer = Normalizer() \
    .setInputCol("tfidf_features") \
    .setOutputCol("norm_features") \
    .setP(2.0)

normalized_df = normalizer.transform(tfidf_df)

print("\n✅ TF-IDF computation complete")
print(f"Feature dimension: 65536")
normalized_df.select("file_name", "norm_features").show(3, truncate=60)

# ------------------------------------------------------------
# STEP 7: Cosine Similarity - FIXED VERSION
# ------------------------------------------------------------
print("\n📐 STEP 7: Calculating cosine similarity...")

# Convert to RDD for easier vector manipulation
print("Converting to RDD for similarity calculation...")
vector_rdd = normalized_df.select("file_name", "norm_features").rdd

# Collect all vectors to driver (OK for small dataset)
vectors = vector_rdd.collect()
print(f"Collected {len(vectors)} vectors")

# Create a dictionary mapping filename to vector
vector_dict = {row.file_name: row.norm_features for row in vectors}

# Choose target book
target_book = "200.txt"
if target_book not in vector_dict:
    print(f"⚠️ {target_book} not found. Using first available book.")
    target_book = list(vector_dict.keys())[0]

target_vector = vector_dict[target_book]
print(f"\n📖 Finding books similar to: {target_book}")

# Calculate cosine similarity for all books
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two MLlib vectors"""
    try:
        # Convert to numpy arrays
        if isinstance(v1, SparseVector):
            arr1 = v1.toArray()
        else:
            arr1 = np.array(v1)
            
        if isinstance(v2, SparseVector):
            arr2 = v2.toArray()
        else:
            arr2 = np.array(v2)
        
        # Dot product (for normalized vectors)
        similarity = float(np.dot(arr1, arr2))
        return similarity
    except Exception as e:
        return 0.0

# Compute similarities
similarities = []
for filename, vector in vector_dict.items():
    if filename != target_book:
        sim = cosine_similarity(target_vector, vector)
        similarities.append((filename, sim))

# Sort by similarity (descending)
similarities.sort(key=lambda x: x[1], reverse=True)

# Get top 5
top_5 = similarities[:5]

print("\n🏆 TOP 5 MOST SIMILAR BOOKS:")
print("-" * 60)
print(f"{'Rank':<6} {'Book':<30} {'Cosine Similarity':<20}")
print("-" * 60)

for i, (book, score) in enumerate(top_5, 1):
    print(f"{i:<6} {book:<30} {score:.6f}")

# Also show all similarities for reference
print("\n📊 All books similarity scores:")
print("-" * 60)
for book, score in similarities:
    print(f"{book:<30} {score:.6f}")

# ------------------------------------------------------------
# STEP 8: Save results
# ------------------------------------------------------------
print("\n💾 STEP 8: Saving results...")

# Save to local file
with open("q11_similarity_results.txt", "w") as f:
    f.write("Top 5 books similar to {}:\n".format(target_book))
    f.write("-" * 50 + "\n")
    for i, (book, score) in enumerate(top_5, 1):
        f.write(f"{i}. {book} - similarity: {score:.6f}\n")
    
    f.write("\n\nAll similarity scores:\n")
    f.write("-" * 50 + "\n")
    for book, score in similarities:
        f.write(f"{book}: {score:.6f}\n")

print("\n✅ Results saved to 'q11_similarity_results.txt'")
print("="*60)
