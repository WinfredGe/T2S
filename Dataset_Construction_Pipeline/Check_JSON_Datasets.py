import os
import re

import os
###########################################################
#                       FIND NULL                         #
###########################################################

def find_missing_ids(directory, prefix="data_sample_", suffix=".json"):
    # Gets all the filenames that match the criteria
    filenames = [filename for filename in os.listdir(directory) if
                 filename.startswith(prefix) and filename.endswith(suffix)]

    # Extract the numeric part of the filename and convert it to an integer
    ids = sorted([filename[len(prefix):-len(suffix)].split('_') for filename in filenames])

    # Convert extracted numbers into tuples (first id, second id)
    id_tuples = [(int(first), int(second)) for first, second in ids]

    # Find the missing labels
    missing_ids = []
    first_ids = set(first for first, _ in id_tuples)

    # Check that each first number has the full 5 files
    for first_id in sorted(first_ids):
        second_ids = {second for _, second in id_tuples if _ == first_id}
        for second_id in range(1, 2):  # check from 1 to 5
            if second_id not in second_ids:
                missing_ids.append((first_id, second_id))

    return missing_ids


# Specifies the directory to store the JSON file
directory = './ETTh1_48_folder/'

# Find the missing ID
missing_ids = find_missing_ids(directory)

# Output the missing ID
if missing_ids:
    print("Missing IDs:")
    for first_id, second_id in missing_ids:
        print(f"{first_id}_{second_id}")
else:
    print("There are no missing files.")


###########################################################
#                       CHECK AVAILABLE                   #
###########################################################



directory = './ETTh1_96_folder/json_data/'

# Total sample ID range
min_sample_id = 1
max_sample_id = 100 #xxx

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
            print(f"sample ID {sample_id} Files with all y-values completely missing.")
        else:
            missing_y = ', '.join(str(y) for y in miss)
            print(f"sample ID {sample_id} missing Y value: {missing_y}")
else:
    print("Files with Y values 1 to 5 already exist for all samples.")


