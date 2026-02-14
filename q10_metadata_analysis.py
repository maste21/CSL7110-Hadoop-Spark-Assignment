#!/usr/bin/env python3
"""
CSL7110 Assignment - Question 10 (FIXED)
Book Metadata Extraction and Analysis
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("BookMetadataAnalysis") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .getOrCreate()

print("="*60)
print("QUESTION 10: Book Metadata Extraction and Analysis")
print("="*60)

# ------------------------------------------------------------
# STEP 1: Load the Gutenberg dataset
# ------------------------------------------------------------
print("\n📂 STEP 1: Loading Gutenberg dataset...")

def load_books(directory_path):
    """Load all .txt files from directory into DataFrame"""
    from pyspark.sql import Row
    
    books = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                books.append(Row(
                    file_name=filename,
                    text=content
                ))
                print(f"  ✓ Loaded: {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
    
    if not books:
        print("❌ No books found! Please check the directory path.")
        return spark.createDataFrame([], schema="file_name STRING, text STRING")
    
    return spark.createDataFrame(books)

# Load the dataset
books_df = load_books("gutenberg_dataset")
print(f"\n✅ Total books loaded: {books_df.count()}")

# Show sample
print("\n📖 Sample books:")
books_df.select("file_name").show(5, truncate=False)

# ------------------------------------------------------------
# STEP 2: Extract metadata using regular expressions
# ------------------------------------------------------------
print("\n🔍 STEP 2: Extracting metadata with regex...")

metadata_df = books_df.select(
    col("file_name"),
    col("text"),
    
    # Extract Title
    regexp_extract(
        col("text"), 
        r'Title:\s*(.+?)(?:\r?\n|$)', 
        1
    ).alias("title"),
    
    # Extract Release Date
    regexp_extract(
        col("text"), 
        r'Release Date:\s*(.+?)(?:\r?\n|$)', 
        1
    ).alias("release_date"),
    
    # Extract Language
    regexp_extract(
        col("text"), 
        r'Language:\s*(.+?)(?:\r?\n|$)', 
        1
    ).alias("language"),
    
    # Extract Encoding
    regexp_extract(
        col("text"), 
        r'Character set encoding:\s*(.+?)(?:\r?\n|$)', 
        1
    ).alias("encoding")
)

print("\n✅ Metadata extracted successfully!")
print("\n📋 Sample extracted metadata:")
metadata_df.select(
    "file_name", "title", "release_date", "language", "encoding"
).show(10, truncate=50)

# ------------------------------------------------------------
# STEP 3: Extract year from release_date (FIXED - handles empty values)
# ------------------------------------------------------------
print("\n📅 STEP 3: Extracting year from release_date...")

# First extract year as string, then filter out empty/non-numeric
books_with_year_str = metadata_df.withColumn(
    "year_str",
    regexp_extract(col("release_date"), r'(\d{4})', 1)
)

# Filter out rows with empty year_str, then cast to int
books_with_year = books_with_year_str.filter(
    col("year_str") != ""
).withColumn(
    "release_year",
    col("year_str").cast(IntegerType())
).drop("year_str")

print(f"✅ Found {books_with_year.filter(col('release_year').isNotNull()).count()} books with valid years")

# ------------------------------------------------------------
# STEP 4: Calculate books released each year
# ------------------------------------------------------------
print("\n📊 STEP 4: Books released per year:")

yearly_counts = books_with_year.filter(
    col("release_year").isNotNull()
).groupBy("release_year").agg(
    count("*").alias("book_count")
).orderBy(col("release_year").desc())

print("\n📅 BOOKS RELEASED PER YEAR:")
print("-" * 40)
yearly_counts.show(20, truncate=False)

# ------------------------------------------------------------
# STEP 5: Find most common language
# ------------------------------------------------------------
print("\n🌐 STEP 5: Most common language:")

language_counts = metadata_df.filter(
    col("language").isNotNull() & 
    (col("language") != "")
).groupBy("language").agg(
    count("*").alias("count")
).orderBy(col("count").desc())

print("\n🌐 MOST COMMON LANGUAGES:")
print("-" * 40)
language_counts.show(10, truncate=False)

# ------------------------------------------------------------
# STEP 6: Calculate average title length
# ------------------------------------------------------------
print("\n📏 STEP 6: Average title length:")

title_stats = metadata_df.filter(
    col("title").isNotNull() & 
    (col("title") != "")
).select(
    col("title"),
    length(col("title")).alias("title_length")
)

avg_title_length = title_stats.select(
    avg("title_length").alias("avg_length")
).collect()[0][0]

print(f"\n📏 Average title length: {avg_title_length:.2f} characters")
print(f"📊 Title length statistics:")
title_stats.describe().show()

# ------------------------------------------------------------
# STEP 7: Save results
# ------------------------------------------------------------
print("\n💾 STEP 7: Saving results...")

metadata_df.select(
    "file_name", "title", "release_date", "language", "encoding"
).write.mode("overwrite").csv("q10_metadata_output")

print("\n✅ Q10 complete! Results saved to 'q10_metadata_output/'")
print("="*60)
