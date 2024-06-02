import csv
import string

def generate_keys(base_key):
    keys = []
    # Iterate over all possible second characters
    for second_char in string.ascii_uppercase:
        # Form the key by appending the first and second characters to the base key
        keys.append(f"{base_key}{second_char}")
    return keys

def main():
    base_key = "GCCKMP"  # Base part of the key

    # Generate keys based on the entered first character
    keys = generate_keys(base_key)

    # Write to CSV file
    with open('keys.csv', 'w', newline='') as csvfile:
        keywriter = csv.writer(csvfile)
        keywriter.writerow(['Key'])  # Write header
        for key in keys:
            keywriter.writerow([key])

if __name__ == "__main__":
    main()
