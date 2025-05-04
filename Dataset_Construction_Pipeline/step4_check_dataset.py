import os
import re

directory = './ETTh1_96_folder/json_data/'

# Total sample ID range
min_sample_id = 1
max_sample_id = xxx

# The regular expression matches the filename pattern
pattern = re.compile(r'data_sample_(\d+)_(\d+)\.json')

# Use a dictionary to store the set of y-values for each sample ID
samples = {}

# Iterate over all files in the directory
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        sample_id = int(match.group(1))
        y_value = int(match.group(2))
        if sample_id not in samples:
            samples[sample_id] = set()
        samples[sample_id].add(y_value)

# Check the sample ID and Y value
missing_samples = {}
missing_y_per_sample = {}

for sample_id in range(min_sample_id, max_sample_id + 1):
    if sample_id not in samples:
        # If the sample ID does not exist, all y-values are missing
        missing_y_per_sample[sample_id] = list(range(1, 2))
    else:
        # Check that the Y value is complete
        missing_y = [y for y in range(1, 2) if y not in samples[sample_id]]
        if missing_y:
            missing_y_per_sample[sample_id] = missing_y

# 输出结果
if missing_y_per_sample:
    print("The following samples are missing corresponding files：")
    for sample_id in sorted(missing_y_per_sample.keys()):
        miss = missing_y_per_sample[sample_id]
        if len(miss) == 5:
            print(f"sample ID {sample id} Files with all y-values completely missing.")
        else:
            missing_y = ', '.join(str(y) for y in miss)
            print(f"sample ID {sample id} missing Y value: {missing y}")
else:
    print("Files with Y values 1 to 5 already exist for all samples.")
