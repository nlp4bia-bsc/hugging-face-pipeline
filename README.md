# 🤗 Hugging Face Pipeline

Automation tools for different processes regarding the training and deployment of named entity recognition (NER) models and datasets to Hugging Face.

NLP for Biomedical Information Analysis (NLP4BIA).

## 💻 Pre-requisites

- Operative System (OS): Linux-based, preferrably Ubuntu 22.04. There are many commands within Python scripts that won't work in other OS, I should work on that.
- Python >= 3.8, though you can try older versions.
- Create a virtual environment and install `requirements.txt`.

## 💪 Training Pipeline

The general pipeline for NER is the following:
1. Annotate using Brat, download the corpus. An example can be found in the [zip](https://zenodo.org/records/8224056/files/medprocner_gs_train+test+gazz+multilingual+crossmap_230808.zip?download=1) file of the [MedProcNER](https://zenodo.org/records/8224056) Shared Task page, in the train and test directories.
2. Make the splits, only **if not already split**, with the `generate_val_split.py` script. Indicate the source (-s), destination (-d), and test_size (number of txts if type integer; percentage if float between 0 and 1)
    - `python hugging-face-pipeline/scripts/generate_val_split.py -s <my-corpus>/train -d <my-corpus>/test --test_size 125`
3. Use the `build_dataset.py` script to create a Hugging Face Dataset with the pre-tokenized input. Indicate the source directory of the corpus (-d) and the output name/path of the Hugging Face dataset (-n).
    - `python hugging-face-pipeline/scripts/build_dataset.py -d <my-corpus>/ -n <my-corpus-ner>`
4. Train the model. Each experiment i.e., a run or combination of hyperparameters, is evaluated on the validation set after each epoch. We apply Early Stopping with 5-epoch patience. There are two alternative scripts:
    - `simple_train.py`: vanilla training for one experiment, where you can pass the hyperparameters directly. The default values should be acceptable for most fine-tunings (100-2000 documents) but don't expect the best results without proper hyperparameter tuning.
    - `train.py` (⚠ **needs internet**): using Weights & Biases Sweep functionality, where you define the list of hyperparameters on a *.yaml* file and you can monitor the training in real-time on wandb.ai. There's an example file in `templates/sweep_default.yaml`. You will need a Weights & Biases account for that (free for academics. After training all experiments, select the best model/experiment using your criteria (e.g. best strict/relaxed F1 on validation).


## 🔍 Inference Pipeline: new method `simple_inference.py`

This method is better suited for large datasets, though it can be slower, than the old method. You can check all arguments in the script. This one only needs the directory of txt files and the model. Example:

- `python/hugging-face-pipeline/scripts/simple_inference.py -i <txt-files> -m <path-to-hf-model> -o <output-dir> -agg first`

### 🔎 Alternative Inference Pipeline: old method `model_inference.py`

We will reuse the Training Pipeline and make inference to the **test** set of the dataset.

1. Generate empty .ann files with: `find /path/to/your/directory -type f -name '*.txt' -exec bash -c 'touch "${1%.txt}.ann"' _ {} \;`. The .ann files are needed for the pre-tokenization that takes place in the `brat2conll` subscript of `build_dataset.py`.
2. Create the same directory structure of training (train/valid/test with all .txts and .anns). We will generate the predictions for the test split only, so train and valid can be just a copy of test.
    - `cd <my-inference-corpus>`
    - `mkdir test`
    - `mv * test`
    - `cp -r test train`
    - `cp -r test valid`
3. Use the `build_dataset.py` script to create a Hugging Face Dataset with the pre-tokenized input. Indicate the source directory of the corpus (-d) and the output name/path of the Hugging Face dataset (-n).
    - `python hugging-face-pipeline/scripts/build_dataset.py -d <my-corpus>/ -n <my-corpus-ner>`
    
    Make sure that the generated HF dataset loader script (e.g. my-corpus-ner.py) has the right tags and in the same order as the one used during training (usually alphabetic)! Otherwise, you will get swapped entities.
4. Make the inference. For that we have two options: running `model_inference.py` or `multiple_model_inference.py`. You can directly use the `multiple_model_inference.py` if you already have the datasets in HF format, provided with the `build_dataset.py`. The `multiple_model_inference.py` is a wrapper for the `model_inference.py`, and is the recommended way if you have to make predictions with many different models in Mare Nostrum (CTE-AMD or MN5, BSC Infrastructure). Otherwise, you can execute the `model_inference.py` script for each model. It will run on GPU if available in both cases. Make sure that both output directories (for ann and conll files, default is the same for both, with -o argument) do not exist (this is to prevent unwanted overwriting). You should specify one different output directory per model. The output directory will contain .conll files. You can remove them if you want. Example usage:
    - `python hugging-face-pipeline/scripts/model_inference.py -ds <my-corpus-ner> -m <my-model> -ocd <my-corpus>/test -o <my-corpus-predictions>`
    
    Or with `multiple_model_inference.py`, you should modify the script accordingly, specifying models and datasets.
    - `python hugging-face-pipeline/scripts/multiple_model_inference.py`

5. In the case you want to join different model's predictions, join all labels by using:
    - `python hugging-face-pipeline/scripts/join_all_labels.py -i <input-root-directory-predictions-to-join> -o <output-directory-with-all-labels-joined>`

## 📁 Directory structure

- requirements.txt -> list of necessary
- brat -> adapted version of original [brat repository](https://github.com/nlplab/brat) files. Necessary tools for some scripts.
- templates -> files used in different scripts as templates.
- scripts:
    - ann2tsv.py -> Create a tsv (with filename) from .ann files
    - build_dataset.py -> generate a HF NER dataset (CoNLL + loader script)
    - brat2conll.py -> Brat (.txt & .ann) files to CoNLL (.conll)
    - model_inference.py -> process a HF NER dataset with a specific model)
    - multiple_model_inference.py -> call model_inference.py for a list of corpus and models.
    - simple_inference.py -> use the HF NER pipeline directly, which is limited
    - train.py -> script called within Weights & Biases Sweep for training (with hyperparameter optimization)
    - join_all_conlls.sh -> generate a {train|validation|test}.conll files by concatenating in order all conll files in each of the 3 subdirectories
    - join_all_labels.py -> merge into a single .ann file, different .ann files of the same set of documents (each .ann containing different labels). Used to merge different model's predictions.
    - generate_val_split.py -> randomnly generate a split
    - generate_brat_colors.py -> generate colors for brat (to be included in visual.conf) for a given set of classes/entities
- notebooks: directory containing useful tools for debugging, partial pipelines and work not polished yet (paths might need an update).
    - load_predictions.ipynb -> snippet to load .ann files into a pandas dataframe
    - model_inference_pipeline -> tests to integrate within HF NER pipeline (not working)
    - conll2ann.ipynb -> .conll & .txt to Standoff (.ann)
    - predictions2conll.ipynb -> model predictions (.json & original .conll) to CoNLL (.conll) (not used anymore, as predictions are directly written to CoNLL files)


**📝 NOTES**

- Make sure to **NOT** include any other (.txt) files in the root directory rather than the raw text for annotation (e.g. a README.txt).

- The **note annotations** (comments) in the (.ann) files will be **ignored** and won't be included in the CoNLL resulting file. These note annotations take the following format:

```
T1	PROCEDIMIENTO 775 794	ecografía abdominal
#1	AnnotatorNotes T1	This is a free-text note annotation associated with T1
T2	PROCEDIMIENTO 1229 1292	TAC abdomino-pélvico realizado con contraste oral e intravenoso
```

## ✉ Contact

Please, for any comment or doubt, you can contact me on: janrodmir \[at] gmail \[dot] com

## 🔗 References

[^1]: Stenetorp, Pontus and Pyysalo, Sampo and Topi\'{c}, Goran and Ohta, Tomoko and Ananiadou, Sophia and Tsujii, Jun'ichi. brat: a Web-based Tool for {NLP}-Assisted Text Annotation. Proceedings of the Demonstrations Session at EACL 2012, April 2012, Association for Computational Linguistics.
