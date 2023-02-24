from typing import List, Tuple
from task_classification.classification import SemevalModel, SemEvalModelArgs
from twibert import TwiBert
import logging
import pandas as pd
import torch

#pretrain twibert model
twibert = TwiBert()
twibert.train()

prefix = 'taskset/'

train_df = pd.read_csv(prefix + 'twi_train.tsv', sep='\t', header=0)


dev_df = pd.read_csv(prefix + 'twi_dev.tsv', sep='\t', header=0)

test_df = pd.read_csv(prefix + 'twi_test_participants.tsv', sep='\t', header=0)


train_df = pd.DataFrame({
    'text': train_df.tweet.replace(r'\n', ' ', regex=True),
    'label':train_df.label
})

from sklearn.model_selection import train_test_split
train, test = train_test_split(train_df, test_size=0.2)



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)



model_args = SemEvalModelArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = True
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-5
#model_args.n_gpu = 2
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 16
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.labels_list = ["negative", "neutral", "positive"]
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.output_dir = "tuned_twiberta"
model_args.best_model_dir = "tuned_twiberta/best_model"

useCuda =  True if torch.cuda.is_available() else False


# Create a TransformerModel
# Create a TransformerModel
model = SemevalModel('roberta', 'twibert', num_labels=3, use_cuda=useCuda, args=model_args)

# Train the model
model.train_model(train, eval_df=test)

predictions, raw_outputs = model.predict(test_df["tweet"].tolist())

test_df["label"] = predictions

#test_df["label"].replace({ 1: "positive", 0: "negative"}, inplace=True)

df = pd.DataFrame({
    'ID': test_df.ID,
    'label':test_df.label
})

df.to_csv('data/submission_twibert.tsv', sep="\t", index=False)

predictions, raw_outputs = model.predict(dev_df["tweet"].tolist())

dev_df["label"] = predictions

#dev_df["label"].replace({ 1: "positive", 0: "negative"}, inplace=True)

gold_df = pd.read_csv(prefix + 'twi_dev_gold_label.tsv', sep='\t', header=0)

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

scores = accuracy_score(gold_df.label.to_list(), dev_df.label.to_list())
print("Scores: ", scores)


