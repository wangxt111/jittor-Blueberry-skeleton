import numpy as np

def print_npz_keys(file_path):
    """
    Given an NPZ file path, this function prints all the keys within it.

    Args:
        file_path (str): The path to the .npz file.
    """
    try:
        with np.load(file_path) as data:
            print(f"Keys in {file_path}:")
            for key in data.keys():
                print(f"- {key}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

print_npz_keys('newdata/train/mixamo/687.npz')

# Example usage:
# Assuming you have an npz file named 'your_file.npz' in the same directory
# or provide the full path to your file.
# print_npz_keys('your_file.npz')