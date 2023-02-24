import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, AdamW, pipeline
import torch
import os
from pathlib import Path

tokenizer = ByteLevelBPETokenizer()
paths = [str(x) for x in Path('dataset/').glob('*.txt')]

tokenizer.train(files=paths, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk', '<mask>'])
os.mkdir('twibert')
tokenizer.save_model('twibert')

tokenizer = RobertaTokenizerFast.from_pretrained("twibert")
#tokenizer.save_model('twibert')


def mlm(tensor):
  rand = torch.rand(tensor.shape)
  mask_arr = (rand < 0.15) * (tensor > 2)
  for i in range(tensor.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    tensor[i, selection] = 4
  return tensor

from tqdm.auto import tqdm
input_ids = []
mask = []
labels = []


for path in tqdm(paths):
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
  sample = tokenizer(lines, max_length=512, padding='max_length', truncation=True,
                     return_tensors='pt')
  labels.append(sample.input_ids)
  mask.append(sample.attention_mask)
  input_ids.append(mlm(sample.input_ids.detach().clone()))


input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)

encodings = {
    'input_ids': input_ids,
    'attention_mask': mask,
    'labels': labels
}


class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __len__(self):
    return self.encodings['input_ids'].shape[0]
    
  def __getitem__(self, i):
    return { key: tensor[i] for key, tensor in self.encodings.items()}


dataset = Dataset(encodings)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)



model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model.to(device)

model.train()



optim = AdamW(model.parameters(), lr=4e-5)



epochs = 10

for epoch in range(epochs):
  loop = tqdm(dataloader, leave=True)
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=mask, 
                  labels=labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch: {epoch}')
    loop.set_postfix(loss=loss.item())

# tokenizer = RobertaTokenizer.from_pretrained('twibert', max_len=512)

model.save_pretrained('twibert')

fill = pipeline('fill-mask', model='twibert', tokenizer='twibert')

from typing import List, Tuple
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import pandas as pd

prefix = 'data/'

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



model_args = ClassificationArgs()
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



# Create a TransformerModel
# Create a TransformerModel
model = ClassificationModel('roberta', 'twibert', num_labels=3, use_cuda=True, args=model_args)

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


