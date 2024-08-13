import os
import shutil
import argparse
import sys


def argparser():
    ap = argparse.ArgumentParser(
        description='Create a new directory containing a new split with the same filenames as another dataset (content of files might differ).')
    ap.add_argument('-s', '--subset',
                    help='Directory containing the subset of .txt and .ann files that the new dataset should contain.', required=True)
    ap.add_argument('-f', '--from_dir',
                    help='Directory with a train subdirectory containing all files (and the ones from the subset)', required=True)
    return ap


# Function to move files
def move_files(original_valid_txts, dst_train, dst_valid):
    for txt_file in original_valid_txts:
        ann_file = txt_file.replace('.txt', '.ann')
        
        src_txt = os.path.join(dst_train, txt_file)
        dst_txt = os.path.join(dst_valid, txt_file)
        if os.path.exists(src_txt):
            shutil.move(src_txt, dst_txt)
        else:
            print(f"File {src_txt} does not exist.")
        src_ann = os.path.join(dst_train, ann_file)
        dst_ann = os.path.join(dst_valid, ann_file)
        if os.path.exists(src_ann):
            shutil.move(src_ann, dst_ann)
        else:
            print(f"File {src_ann} does not exist.")


def main(argv):
    args = argparser().parse_args(argv[1:])
    original_valid = args.subset
    base_dir = args.from_dir
    dst_train = os.path.join(base_dir, 'train')
    dst_valid = os.path.join(base_dir, 'valid')
    # File lists from the original splits
    original_valid_txts = [filename for filename in os.listdir(original_valid) if filename.endswith('.txt')]
    
    # Create directory if not existing
    if not os.path.exists(dst_valid):
        os.makedirs(dst_valid)
        
    # Move files to valid and test directories
    move_files(original_valid_txts, dst_train, dst_valid)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
