### CSL7110-Hadoop-Spark-Assignment

├── WordCount.java
├── WordCountWithTime.java # Hadoop Q9 - With execution time
├── q10_metadata_analysis.py # Spark Q10 - Book metadata extraction
├── q11_tfidf_similarity.py # Spark Q11 - TF-IDF and similarity
├── q12_author_network.py # Spark Q12 - Author influence network
└── lyrics.txt # Sample input file


### Prerequisites
- Hadoop 3.3.6 installed at /usr/local/hadoop
- Java 17+ installed
- HDFS running

### Compile and Run code
```bash
# Compile
javac -cp $(hadoop classpath) {java_file_name}.java
jar -cvf {jar_name}.jar *.class


hdfs dfs -mkdir -p /user/arun/{q_number}
hdfs dfs -put {file} /user/arun/{q_number}/
hadoop jar {jar_name}.jar {java_file_name} /user/arun/{q_number}/{file}/user/arun/{q_number}/output

#Word count
hdfs dfs -cat /user/arun/q1/output/part-r-00000



### Prerequisites
Spark 4.0.1 installed
PySpark installed (pip install pyspark)
Gutenberg dataset in gutenberg_dataset/ folder


python3 q10_metadata_analysis.py

python3 q11_tfidf_similarity.py

python3 q12_author_network.py
