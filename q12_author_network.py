#!/usr/bin/env python3
"""
CSL7110 Assignment - Question 12
Author Influence Network
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("AuthorInfluenceNetwork") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .getOrCreate()

print("="*60)
print("QUESTION 12: Author Influence Network")
print("="*60)

# ------------------------------------------------------------
# STEP 1: Load books and extract metadata
# ------------------------------------------------------------
print("\n📂 STEP 1: Loading books and extracting metadata...")

def load_books_with_metadata(directory_path):
    """Load books and extract author and release year"""
    from pyspark.sql import Row
    
    books = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:10000]  # First 10k chars for metadata
                
                # Extract Author using regex
                import re
                author_match = re.search(r'Author:\s*(.+?)(?:\r?\n|$)', content)
                author = author_match.group(1).strip() if author_match else None
                
                # Clean author (remove "by", dates in parentheses, etc.)
                if author:
                    author = re.sub(r'^by\s+', '', author, flags=re.IGNORECASE)
                    author = re.sub(r'\s*\(\d{4}[^)]*\)', '', author)
                    author = re.sub(r'[,\\.;]$', '', author).strip()
                
                # Extract Release Date
                date_match = re.search(r'Release Date:\s*(.+?)(?:\r?\n|$)', content)
                date_str = date_match.group(1).strip() if date_match else None
                
                # Extract year
                year = None
                if date_str:
                    year_match = re.search(r'(\d{4})', date_str)
                    year = int(year_match.group(1)) if year_match else None
                
                if author and year:
                    books.append(Row(
                        file_name=filename,
                        author_original=author,
                        release_year=year
                    ))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return spark.createDataFrame(books)

# Load dataset
authors_df = load_books_with_metadata("gutenberg_dataset")
print(f"✅ Loaded {authors_df.count()} books with author/year metadata")

print("\n📋 Sample books with authors and years:")
authors_df.show(10, truncate=False)

# ------------------------------------------------------------
# STEP 2: Normalize author names for matching
# ------------------------------------------------------------
print("\n🔧 STEP 2: Normalizing author names...")

# Create normalized author key (lowercase, no punctuation)
authors_clean = authors_df.select(
    col("file_name"),
    col("author_original"),
    col("release_year"),
    # Create normalized key for matching
    regexp_replace(
        lower(col("author_original")), 
        "[^a-z0-9\\s]", 
        ""
    ).alias("author_key")
).select(
    "*",
    regexp_replace(col("author_key"), "\\s+", " ").alias("author_norm")
).drop("author_key")

print("✅ Author names normalized")
authors_clean.show(5, truncate=False)

# Show unique authors
print("\n📊 Unique authors after normalization:")
authors_clean.groupBy("author_norm").agg(
    count("*").alias("book_count"),
    collect_list("author_original").alias("name_variants")
).orderBy(col("book_count").desc()).show(10, truncate=False)

# ------------------------------------------------------------
# STEP 3: Construct influence network for different time windows
# ------------------------------------------------------------
print("\n🔗 STEP 3: Constructing influence network...")

# Self-join to find influence relationships
a1 = authors_clean.alias("a1")
a2 = authors_clean.alias("a2")

# Test different time windows as requested in assignment
time_windows = [1, 5, 10, 20]
window_results = []

print("\n📊 INFLUENCE NETWORK ANALYSIS FOR DIFFERENT TIME WINDOWS:")
print("="*70)

for X in time_windows:
    print(f"\n📈 TIME WINDOW: {X} years")
    print(f"    (Author A influences Author B if A's book released at least {X} years before B)")
    print("-" * 50)
    
    # Create edges: author1 -> author2 if year1 <= year2 - X
    edges = a1.join(
        a2,
        (col("a1.author_norm") != col("a2.author_norm")) &  # Different authors
        (col("a1.release_year") <= col("a2.release_year") - X) &  # Time window condition
        col("a1.release_year").isNotNull() & 
        col("a2.release_year").isNotNull(),
        "inner"
    ).select(
        col("a1.author_original").alias("author_from"),
        col("a2.author_original").alias("author_to"),
        col("a1.release_year").alias("year_from"),
        col("a2.release_year").alias("year_to"),
        (col("a2.release_year") - col("a1.release_year")).alias("year_diff")
    ).distinct()
    
    edge_count = edges.count()
    print(f"📊 Total influence edges: {edge_count}")
    
    # ------------------------------------------------------------
    # STEP 4: Calculate in-degree and out-degree
    # ------------------------------------------------------------
    
    # Out-degree: number of authors this author influenced
    out_degree = edges.groupBy("author_from") \
        .agg(count("*").alias("out_degree")) \
        .orderBy(col("out_degree").desc())
    
    # In-degree: number of authors who influenced this author
    in_degree = edges.groupBy("author_to") \
        .agg(count("*").alias("in_degree")) \
        .orderBy(col("in_degree").desc())
    
    # Get top 5 for each
    top_out = out_degree.limit(5).collect()
    top_in = in_degree.limit(5).collect()
    
    print("\n🏆 TOP 5 INFLUENTIAL AUTHORS (Highest Out-Degree):")
    print(f"{'Rank':<6} {'Author':<40} {'Out-Degree':<12}")
    print("-" * 60)
    for i, row in enumerate(top_out, 1):
        author = row['author_from'][:38] + ".." if len(row['author_from']) > 38 else row['author_from']
        print(f"{i:<6} {author:<40} {row['out_degree']:<12}")
    
    print("\n🏆 TOP 5 INFLUENCED AUTHORS (Highest In-Degree):")
    print(f"{'Rank':<6} {'Author':<40} {'In-Degree':<12}")
    print("-" * 60)
    for i, row in enumerate(top_in, 1):
        author = row['author_to'][:38] + ".." if len(row['author_to']) > 38 else row['author_to']
        print(f"{i:<6} {author:<40} {row['in_degree']:<12}")
    
    # Store results
    window_results.append({
        'window': X,
        'edges': edge_count,
        'top_out': top_out,
        'top_in': top_in
    })
    
    # Save for X=5 as requested
    if X == 5:
        edges.write.mode("overwrite").csv("q12_edges_output")
        out_degree.write.mode("overwrite").csv("q12_out_degree_output")
        in_degree.write.mode("overwrite").csv("q12_in_degree_output")

# ------------------------------------------------------------
# STEP 5: Show impact of time window on network size
# ------------------------------------------------------------
print("\n📈 STEP 5: Impact of time window on network size:")
print("="*60)
print(f"{'Window (years)':<16} {'Number of Edges':<20} {'Change from X=5':<20}")
print("-" * 60)

base_edges = [r['edges'] for r in window_results if r['window'] == 5][0]
for result in window_results:
    change = (result['edges'] / base_edges) if base_edges > 0 else 0
    print(f"{result['window']:<16} {result['edges']:<20} {change:.2f}x")

print("\n✅ Q12 complete! Results saved to:")
print("  - q12_edges_output/")
print("  - q12_out_degree_output/")
print("  - q12_in_degree_output/")
print("="*60)
