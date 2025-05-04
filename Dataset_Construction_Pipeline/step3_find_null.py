import os


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
