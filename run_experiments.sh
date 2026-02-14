#!/bin/bash
echo "==================================================================="
echo "     QUESTION 9: SPLIT SIZE PERFORMANCE EXPERIMENT"
echo "==================================================================="
echo ""

# Experiment 1: Default split size (128MB)
echo "EXPERIMENT 1: DEFAULT SPLIT SIZE (128 MB)"
echo "-------------------------------------------------------------------"
hadoop jar WordCountWithTime.jar WordCountWithTime /user/arun/q9/large_200.txt /user/arun/q9/output_default
echo ""

# Experiment 2: Small split size (16MB = 16777216 bytes)
echo "EXPERIMENT 2: SMALL SPLIT SIZE (16 MB)"
echo "-------------------------------------------------------------------"
hadoop jar WordCountWithTime.jar WordCountWithTime /user/arun/q9/large_200.txt /user/arun/q9/output_small 16777216
echo ""

# Experiment 3: Large split size (256MB = 268435456 bytes)
echo "EXPERIMENT 3: LARGE SPLIT SIZE (256 MB)"
echo "-------------------------------------------------------------------"
hadoop jar WordCountWithTime.jar WordCountWithTime /user/arun/q9/large_200.txt /user/arun/q9/output_large 268435456
echo ""

# Experiment 4: Very Small split size (4MB = 4194304 bytes)
echo "EXPERIMENT 4: VERY SMALL SPLIT SIZE (4 MB)"
echo "-------------------------------------------------------------------"
hadoop jar WordCountWithTime.jar WordCountWithTime /user/arun/q9/large_200.txt /user/arun/q9/output_verysmall 4194304
echo ""

echo "==================================================================="
echo "                        EXPERIMENTS COMPLETE"
echo "==================================================================="
