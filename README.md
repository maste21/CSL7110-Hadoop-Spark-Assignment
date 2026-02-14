### CSL7110-Hadoop-Spark-Assignment

<img width="1017" height="314" alt="image" src="https://github.com/user-attachments/assets/a8801c02-65aa-4f7e-b274-9f6ac6c9685a" />



### Prerequisites
- Hadoop 3.3.6 installed at /usr/local/hadoop
- Java 17+ installed
- HDFS running

- Spark 4.0.1 installed
- PySpark installed (pip install pyspark)
- Gutenberg dataset in gutenberg_dataset/ folder

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


python3 q10_metadata_analysis.py

python3 q11_tfidf_similarity.py

python3 q12_author_network.py
