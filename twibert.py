import torch
from tqdm.auto import tqdm
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, AdamW, pipeline
from dataset import Dataset


# based on:
# https://huggingface.co/blog/how-to-train

class TwiBert:
    def __init__(
        self, 
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        learning_rate = 4e-5,
        epochs = 5,
        batch_size = 16
        ):

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
      

        # define token

        tokenizer = ByteLevelBPETokenizer()
        if not os.path.exists('data'):
            os.mkdir('data')

        paths = [str(x) for x in Path('data/').glob('*.txt')]
        self.paths = paths
        tokenizer.train(files=paths, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk', '<mask>'])
        
        
        if not os.path.exists('twibert'):
            os.mkdir('twibert')
        tokenizer.save_model('twibert')

        self.tokenizer = RobertaTokenizerFast.from_pretrained("twibert")



    def mlm(self,tensor):
        rand = torch.rand(tensor.shape)
        mask_arr = (rand < 0.15) * (tensor > 2)
        for i in range(tensor.shape[0]):
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            tensor[i, selection] = 4
        return tensor
    
    def _prepare_encodings(self, paths):
        
        input_ids = []
        mask = []
        labels = []

        for path in tqdm(paths):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')
            sample = self.tokenizer(lines, max_length=512, padding='max_length', truncation=True,
                     return_tensors='pt')
            labels.append(sample.input_ids)
            mask.append(sample.attention_mask)
            input_ids.append(self.mlm(sample.input_ids.detach().clone()))
        
        input_ids = torch.cat(input_ids)
        mask = torch.cat(mask)
        labels = torch.cat(labels)
        encodings = {
            'input_ids': input_ids,
            'attention_mask': mask,
            'labels': labels
            }
        return encodings
        


    def train(self):
        encodings = self._prepare_encodings(self.paths)
        dataset = Dataset(encodings)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            type_vocab_size=self.type_vocab_size
            )
        
        model = RobertaForMaskedLM(config)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        model.train()


        optim = AdamW(model.parameters(), lr=self.learning_rate)
        
        epochs = self.epochs
        
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

        model.save_pretrained('twibert')






