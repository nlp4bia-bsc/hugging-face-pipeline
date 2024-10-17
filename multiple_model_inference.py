import sys
import os
from time import gmtime, strftime
from model_inference import main as model_inference_main

# Define your models and other arguments
models_base_dir = "/gpfs/projects/bsc14/MN4/bsc14/hugging_face_models/ner" # no trailing '/'
models = [
    "bsc-bio-ehr-es-meddoprof-no-act-es",
    "bsc-bio-ehr-es-carmen-distemist",
    "bsc-bio-ehr-es-carmen-medprocner",
    "bsc-bio-ehr-es-meddoplace",
    "bsc-bio-ehr-es-meddoplace-subclasses",
    "bsc-bio-ehr-es-carmen-symptemist",
    "bsc-bio-ehr-es-carmen-livingner-species",
    "bsc-bio-ehr-es-carmen-livingner-humano",
    "bsc-bio-ehr-es-carmen-drugtemist",
    "bsc-bio-ehr-es-carmen-meddocan",
]

# Define the datasets in a tuple (-ds, -ocd), referring to the "-ner" dataset and the original conll dir.
datasets = [
    ("mesinesp2-literature-ner", "mesinesp2-literature-txts/test")
]

# Iterate over models and call the main() function directly
for ds, ocd in datasets:
    output_base_dir = f"/gpfs/scratch/bsc14/MN4/bsc14/bsc14527/predictions/{ds}_{strftime('%Y%m%d_%H%M%S', gmtime())}"
    os.makedirs(output_base_dir, exist_ok=False)
    for model in models:
        print(f"Starting to process {ds} dataset with {model}.")
        model_path = models_base_dir + '/' + model
        # Define the output directory for this model's results
        output_anns_dir = f"{output_base_dir}/{model}-{ds}"
        
        # Build the argument list, simulating command-line arguments
        argv = [
            "model_inference.py",  # This mimics sys.argv[0], which is the script name
            "-ds", ds,
            "-m", model_path,
            "-ocd", ocd,
            "-o", output_anns_dir
        ]
    
        # Call the main function from model_inference.py with the arguments
        try:
            model_inference_main(argv)
            print(f"Completed inference for {model} on dataset {ds}.")
        except Exception as e:
            print(e)
            print(f"Failed to run inference for {model} on dataset {ds}")
    print(f"Dataset {ds} processed.")
