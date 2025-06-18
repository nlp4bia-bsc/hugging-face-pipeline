import os
import sys
import argparse

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process directories and combine .ann files.")
    # Adding two positional arguments for root and output directories
    parser.add_argument('-i', '--root_dir', type=str, help='Input root directory containing prediction folders.')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory to save combined annotation files.')
    return parser


# Function to join all labels from .ann files in the root directory
def join_all_labels(root_dir, output_dir):
    # Get the list of directories in the root directory
    dirs = [dir for dir in os.scandir(root_dir)]
    
    # Get the list of .ann documents in the first directory
    documents = [doc for doc in os.listdir(dirs[0].path) if doc.endswith('.ann')]
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each document
    for i, doc in enumerate(documents):
        lines = []
        # Iterate through all subdirectories in the root directory
        for dir in dirs:
            # Check if the document exists in the current subdirectory
            if doc in os.listdir(dir.path):
                # Read and accumulate the lines from the document
                with open(os.path.join(dir.path, doc), 'r') as d:
                    lines += d.readlines()
        
        # Writing combined lines to the output file
        num = 1
        with open(os.path.join(output_dir, doc), "a") as d2:
            for j, line in enumerate(lines):
                if not line.endswith("\n") and j != len(lines)-1:
                    line += "\n" # add this to ensure entities across diferent annotation models are not concatenated
                # Split the line to format the output properly
                _line = line.split("\t")
                d2.write("T{}\t{}\t{}".format(num, _line[1], '\t'.join(_line[2:])))
                num += 1
    print(f"Processed {i+1}/{len(documents)}.")
        

# Main function to run the script
def main(argv):
    # Parse command line arguments
    args = parse_args().parse_args(argv[1:])
    # Call the function to join labels, passing the root and output directories
    join_all_labels(args.root_dir, args.output_dir)

# Entry point for the script
if __name__ == '__main__':
    sys.exit(main(sys.argv))
