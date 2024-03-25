#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning the model
# Author: Jan Rodríguez Miret
# 
# ### Resources
# - https://huggingface.co/docs/transformers/main/tasks/token_classification
# - https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb

# In[ ]:


#!python -m pip install transformers==4.30.1 evaluate datasets seqeval accelerate wandb nervaluate plotly


# In[2]:


import os
import transformers
import pandas as pd
import numpy as np
import torch
import wandb
import time
from datasets import load_dataset
from nervaluate import Evaluator
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report


def train(config):
  
  with open("wandb_key.txt", 'r') as key_file:
    wandb.login(key=key_file.read())
  with wandb.init(config=config):
    config = wandb.config
    
    # Environment variables
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512" # otherwise we get HIP Error for memory fragmentation
    os.environ["WANDB_NOTEBOOK_NAME"] = f"{config.PROJECT_NAME.replace('-','_')}.ipynb"

    # General
    PROJECT_NAME = config.PROJECT_NAME
    HF_DATASET = config.HF_DATASET
    BASE_MODEL = config.BASE_MODEL
    
    # Training args
    _weight_strategy = config._weight_strategy
    _num_epochs = config._num_epochs
    _batch_size = config._batch_size
    _learning_rate = config._learning_rate
    _evalutation_strategy = config._evalutation_strategy
    _weight_decay = config._weight_decay
    _output_dir = config._output_dir
    _train_ratio_for_eval = config._train_ratio_for_eval
    _warmup_ratio = config._warmup_ratio
    _classifier_dropout = config._classifier_dropout
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    
    
    # In[10]:
    
    
    dataset = load_dataset(HF_DATASET)
    dataset
    
    
    # ## Training
    
    # ### Prepare data & class weights
    
    # In[11]:
    
    
    classes = dataset["train"].features["ner_tags"].feature
    id2label = {idx: tag for idx, tag in enumerate(classes.names)}
    label2id = {tag: idx for idx, tag in enumerate(classes.names)}
    
    
    # In[12]:
    
    
    def create_tag_names(batch):
      return {"ner_tags_str": [id2label[x] for x in batch["ner_tags"]]}
    
    dataset_str = dataset.map(create_tag_names)
    
    
    # In[13]:
    
    
    from collections import Counter, defaultdict
    
    
    split2freqs = defaultdict(Counter)
    
    # Iterate for each ner_tag, of each sample, of each subset split
    for split_name, dataset_split in dataset_str.items():
      for row in dataset_split["ner_tags_str"]:
        for tag in row:
            split2freqs[split_name][tag] +=1
    
    
    freqs = pd.DataFrame.from_dict(split2freqs, orient="index")
    freqs
    
    
    # In[14]:
    
    
    from math import sqrt
    
    total_weight = sum(split2freqs['train'].values())
    print("Total sum weight:", total_weight)
    
    class_weights_none = pd.DataFrame([[key for key in split2freqs["train"]], [1.0 for freq in split2freqs["train"].values()]]).T.set_index(0)
    class_weights_freq = pd.DataFrame([[key for key in split2freqs["train"]], [total_weight/(freq* classes.num_classes) for freq in split2freqs["train"].values()]]).T.set_index(0)
    class_weights_freq_sqrt = pd.DataFrame([[key for key in split2freqs["train"]], [sqrt(total_weight/(freq* classes.num_classes)) for freq in split2freqs["train"].values()]]).T.set_index(0)
    
    sum_weight_sqrt = sum([split2freqs['train'][key] * class_weights_freq[1].apply(sqrt).loc[key] for key in split2freqs['train']])
    print("Total sum square rooted weight:", sum_weight_sqrt)
    
    # We can use this factor to adapt the learning rate, so that it learns at the same pace as before (total_weights * lr = constant)
    print("Ratio:", total_weight / sum_weight_sqrt)
    
    
    if _weight_strategy == 'none':
        class_weights = class_weights_none
    elif _weight_strategy == 'freq':
        class_weights = class_weights_freq
    elif _weight_strategy == 'freq_sqrt':
        class_weights = class_weights_freq_sqrt
    
    # Reorder class names with the same as label ids (class 0, 1, 2, 3)
    # In case the IOB label is not present in the training set, just add an arbitrary number.
    # It should not affect the loss computation, as it will never encounter this label during training
    class_weights = [class_weights.loc[lab].values[0] if (lab in class_weights.index) else 1. for lab in classes.names]
    print(class_weights_none)
    print(class_weights_freq)
    print(class_weights_freq_sqrt)
    print(f"Selected weights ('{_weight_strategy}'):", class_weights)
    
    
    # In[15]:
    
    
    from transformers import AutoTokenizer
    
    # checkpoint = "bsc-bio-ehr-es-meddoplace/checkpoint-3598"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer
    
    
    # In[16]:
    
    
    def tokenize_and_align_labels(samples):
        tokenized_inputs = tokenizer(samples["tokens"], truncation=True, is_split_into_words=True)
    
        labs = []
        for i, label in enumerate(samples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to the current label
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
    
            labs.append(label_ids)
    
        tokenized_inputs["labels"] = labs
        return tokenized_inputs
    
    
    # In[17]:
    
    
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset
    
    
    # ### Evaluation metrics
    
    # In[18]:
    
    
    def align_predictions(predictions, label_ids):
      # Obtenemos prediccionnes
      preds = np.argmax(predictions, axis=2)
      batch_size, seq_len = preds.shape
      labels_list, preds_list = [], []
    
      #  en cada batch de datos
      for batch_idx in range(batch_size):
        # Generamos muestras de true_values y predicciones en formato seq-eval
        example_labels, example_preds = [], []
        # Para cada documento
        for seq_idx in range(seq_len):
          # Ignoramos etiquetas que sean -100
          if label_ids[batch_idx, seq_idx] != -100:
            example_labels.append(id2label[label_ids[batch_idx][seq_idx]])
            example_preds.append(id2label[preds[batch_idx][seq_idx]])
    
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    
      return preds_list, labels_list
    
    
    # In[19]:
    
    
    # Jan's version for computing relaxed matches
    # Five categories: exact_w_same_class, some_ov_w_same_class, exact_w_other_class, some_ov_w_other_class, complete_spurious
    def get_entity_class(bio_label):
        if bio_label == 'O':
            return bio_label
        return bio_label.split('-')[1]
    
    def get_mentions(tags):
        mentions = []
        start_idx = None
        last_class = None
    
        for i, tag in enumerate(tags):
            if tag == 'O':
                if start_idx is not None:
                    mentions.append((pd.Interval(start_idx, i-1, closed='both'), last_class))
                start_idx = None
                last_class = None
                continue
            class_prefix, class_label = tag.split('-')
    
            if class_prefix == 'B':
                if start_idx is not None:
                    mentions.append((pd.Interval(start_idx, i-1, closed='both'), last_class))
                start_idx = i
                last_class = class_label
            else:
                if start_idx is None or last_class != class_label:
                    if start_idx is not None:
                        mentions.append((pd.Interval(start_idx, i-1, closed='both'), last_class))
                    start_idx = i
                    last_class = class_label
    
        if start_idx is not None:
            mentions.append((pd.Interval(start_idx, len(tags) - 1, closed='both'), last_class))
    
        return mentions
    
    import copy
    
    def compute_relaxed_matches(y_true, y_pred, fp_or_fn='fp'):
        # FP analysis (by default)
        # NOTE that if FN, y_pred and y_true get swapped in terms of names!
        if fp_or_fn == 'fn':
            y_true_aux = copy.deepcopy(y_true)
            y_true = copy.deepcopy(y_pred)
            y_pred = y_true_aux
        all_exact_w_same_class = []
        all_some_ov_w_same_class = []
        all_exact_w_other_class = []
        all_some_ov_w_other_class = []
        all_complete_spurious = []
        for sent_idx, (y_true_sent, y_pred_sent) in enumerate(zip(y_true, y_pred)):
            mentions_example_true = get_mentions(y_true_sent)
            mentions_example_pred = get_mentions(y_pred_sent)
            for pred in mentions_example_pred:
                exact_w_same_class = False
                some_ov_w_same_class = False
                exact_w_other_class = False
                some_ov_w_other_class = False
                complete_spurious = False
                for true in mentions_example_true:
                    if pred[0] == true[0] and pred[1] == true[1]:
                        exact_w_same_class = True
                    elif (pred[0].left in true[0] or pred[0].right in true[0]) and pred[1] == true[1]:
                        some_ov_w_same_class = True
                    elif pred[0] == true[0] and pred[1] != true[1]:
                        exact_w_other_class = True
                    elif (pred[0].left in true[0] or pred[0].right in true[0]) and pred[1] != true[1]:
                        some_ov_w_other_class = True
                # Add predicted mention only to the best match
                if exact_w_same_class:
                    all_exact_w_same_class.append((sent_idx, pred[0], pred[1]))
                elif some_ov_w_same_class:
                    all_some_ov_w_same_class.append((sent_idx, pred[0], pred[1]))
                elif exact_w_other_class:
                    all_exact_w_other_class.append((sent_idx, pred[0], pred[1]))
                elif some_ov_w_other_class:
                    all_some_ov_w_other_class.append((sent_idx, pred[0], pred[1]))
                else:
                    all_complete_spurious.append((sent_idx, pred[0], pred[1]))
        counts = dict(
            exact_w_same_class = len(all_exact_w_same_class),
            some_ov_w_same_class = len(all_some_ov_w_same_class),
            exact_w_other_class = len(all_exact_w_other_class),
            some_ov_w_other_class = len(all_some_ov_w_other_class),
            complete_spurious = len(all_complete_spurious),
        )
        match_types = ['exact_w_same_class', 'some_ov_w_same_class', 'exact_w_other_class', 'some_ov_w_other_class', 'complete_spurious']
        counts_list = [counts[match_type] for match_type in match_types]
        precision_or_recall = np.cumsum(counts_list) / sum(counts_list)
        if fp_or_fn == 'fp':
            precision = {match_types[i]: precision_or_recall[i] for i in range(len(match_types))}
            return { 'counts': counts, 'precision': precision}
        else:
            recall = {match_types[i]: precision_or_recall[i] for i in range(len(match_types))}
            return { 'counts': counts, 'recall': recall}
    
    
    # In[20]:
    
    
    import evaluate
    
    seqeval = evaluate.load("seqeval", experiment_id=wandb.run.id)
    
    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(eval_pred.predictions,
                                           eval_pred.label_ids)
        # Strict match
        results = seqeval.compute(predictions=y_pred, references=y_true)
        # MUC-5 with nervaluate
        evaluator = Evaluator(y_true, y_pred, set([get_entity_class(name) for name in classes.names]), loader='list')
        muc_5_results, muc_5_results_by_tag = evaluator.evaluate()
        # FP & FN relaxed matches (Jan's version)
        fp_analysis = compute_relaxed_matches(y_true, y_pred, fp_or_fn='fp')
        fn_analysis = compute_relaxed_matches(y_true, y_pred, fp_or_fn='fn')
        f1_analysis = { match_type: (2 * fp_analysis['precision'][match_type] * fn_analysis['recall'][match_type]) / \
         (fp_analysis['precision'][match_type] + fn_analysis['recall'][match_type]) for match_type in fp_analysis['precision']}
        wandb.log({})
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "muc_5": muc_5_results,
            "muc_5_by_tag": muc_5_results_by_tag,
            "fp_analysis": fp_analysis,
            "fn_analysis": fn_analysis,
            "f1_analysis": f1_analysis,
        }
    
    
    # ### Prepare model
    
    # In[21]:
    
    
    from transformers import AutoConfig
    
    roberta_config = AutoConfig.from_pretrained(BASE_MODEL,
                                                num_labels = classes.num_classes,
                                                id2label = id2label,
                                                label2id = label2id,
                                                classifier_dropout = _classifier_dropout
                                                )
    
    
    # In[22]:
    
    
    from transformers import RobertaForTokenClassification
    
    model = RobertaForTokenClassification.from_pretrained(BASE_MODEL, config=roberta_config)
    # model = RobertaForTokenClassification.from_pretrained('PlanTL-GOB-ES/bsc-bio-ehr-es-cantemist', config=roberta_config)
    # model = RobertaForTokenClassification.from_pretrained('PlanTL-GOB-ES/bsc-bio-ehr-es', config=roberta_config)
    
    
    # In[23]:
    
    
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=_output_dir,
        num_train_epochs=_num_epochs,
        per_device_train_batch_size=_batch_size,
        per_device_eval_batch_size=_batch_size,
        learning_rate=_learning_rate,
        weight_decay=_weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        # eval_steps=200,
        save_strategy="epoch",
        #push_to_hub=True,
        # overwrite_output_dir=True,
        report_to='wandb',
        warmup_ratio=_warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
    )
    
    
    # In[24]:
    
    
    # Compute and log training metrics
    from transformers import TrainerCallback
    import copy
    
    class ComputeTrainingMetricsCallback(TrainerCallback):
    
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
    
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = copy.deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset.select(range(int(len(self._trainer.train_dataset)*_train_ratio_for_eval))), metric_key_prefix="train")
                return control_copy
    
    
    # In[25]:
    
    
    from transformers import DataCollatorForTokenClassification
    
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')
    
    
    # In[26]:
    
    
    # Define custom loss for class weigths
    from transformers import Trainer
    from torch import nn
    
    class ClassWeightTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    
    # In[27]:
    
    
    trainer = ClassWeightTrainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(ComputeTrainingMetricsCallback(trainer))
    trainer.model
    
    
    # In[28]:

    import time
    
    start_time = time.time()
    #get_ipython().run_cell_magic('time', '', '#train_results = trainer.train(resume_from_checkpoint=True)\ntrain_results = trainer.train()\n')
    train_results = trainer.train()
    end_time = time.time()
    print(end_time - start_time)
    
    trainer.save_model(output_dir=f"{_output_dir}/best-{wandb.run.id}")
    
    # In[29]:
    
    
    train_log = pd.DataFrame([log for log in trainer.state.log_history if 'loss' in log])
    val_log = pd.DataFrame([log for log in trainer.state.log_history if 'eval_loss' in log])
    val_log
    
    
    # In[30]:
    
    
    train_log
    
    
    # In[31]:
    
    
    print("Best model:", trainer.state.best_model_checkpoint)
    

    wandb.run.finish()

if __name__ == '__main__':
    default_config = {
        # General
        "PROJECT_NAME": "drugtemist-es",
        "HF_DATASET": "drugtemist-es-ner",
        "BASE_MODEL": "PlanTL-GOB-ES/bsc-bio-ehr-es",
        
        # Training args
        "_weight_strategy": 'none', # one of ('none', 'freq', 'freq_sqrt')
        "_num_epochs": 20,
        "_batch_size": 16,
        "_learning_rate": 3e-5,
        "_evalutation_strategy": "epoch",
        "_weight_decay": 0.03,
        "_output_dir": f"bsc-bio-ehr-es-drugtemist-es",
        "_train_ratio_for_eval": 1.0,
        "_warmup_ratio": 0.1,
        "_classifier_dropout": 0.5,
    }
    train(default_config)
