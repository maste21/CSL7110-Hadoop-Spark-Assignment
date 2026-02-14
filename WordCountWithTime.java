import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountWithTime {

    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().replaceAll("[^a-zA-Z0-9\\s]", "");
            StringTokenizer tokenizer = new StringTokenizer(line);
            while (tokenizer.hasMoreTokens()) {
                word.set(tokenizer.nextToken());
                context.write(word, one);
            }
        }
    }

    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        if (args.length > 2) {
            long splitSize = Long.parseLong(args[2]);
            conf.setLong("mapreduce.input.fileinputformat.split.maxsize", splitSize);
            System.out.println("=========================================");
            System.out.println("SPLIT SIZE SET TO: " + splitSize + " bytes (" + (splitSize/1024/1024) + " MB)");
            System.out.println("=========================================");
        } else {
            System.out.println("=========================================");
            System.out.println("USING DEFAULT SPLIT SIZE (128 MB)");
            System.out.println("=========================================");
        }
        
        Job job = Job.getInstance(conf, "word count with time");
        job.setJarByClass(WordCountWithTime.class);
        job.setMapperClass(Map.class);
        job.setCombinerClass(Reduce.class);
        job.setReducerClass(Reduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        long startTime = System.currentTimeMillis();
        boolean success = job.waitForCompletion(true);
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        
        System.out.println("\n**************************************************");
        System.out.println("***                                            ***");
        System.out.println("***        JOB EXECUTION TIME: " + executionTime + " ms        ***");
        System.out.println("***                                            ***");
        System.out.println("**************************************************\n");
        
        System.exit(success ? 0 : 1);
    }
}
