import torch
import os
from constants import WORD_START, SPECIAL_CHARS
from torch.utils.data import DataLoader
from torch.nn import Embedding
from viet_diacritic_mark import src_char_to_ix, label_to_ix
from torch_datasets import ViCharDiacDataset
from transformers import DistilBertForTokenClassification, DistilBertTokenizer, Trainer, TrainingArguments
from evaluation import evaluate_accuracy_torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('beginning to load data')
train_dataset = ViCharDiacDataset('data/vi/src_train.txt', 'data/vi/target_train.txt')
print('loaded train')
val_dataset = ViCharDiacDataset('data/vi/src_dev.txt', 'data/vi/target_dev.txt')
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print('loaded val')

def train(epochs, lr, warmup_steps):
  model = DistilBertForTokenClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(label_to_ix))
  dbtokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
  new_embeds = Embedding(len(src_char_to_ix), model.get_input_embeddings().embedding_dim)
  with torch.no_grad():
    for char in src_char_to_ix:
      if char not in SPECIAL_CHARS:
        new_embeds.weight[src_char_to_ix[char]] = model.distilbert.embeddings.word_embeddings.weight[dbtokenizer.encode(char, add_special_tokens=False)[0]].clone()
  model.set_input_embeddings(new_embeds)
  model.distilbert.resize_token_embeddings(len(src_char_to_ix))

  model.to(device)

  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=epochs,         # total number of training epochs
    learning_rate=lr,
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=warmup_steps,       # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
  )

  trainer.train()
  model.eval()
  return model, trainer

def hyperparameter_search():
  max_word_acc = 0
  for lr in [1e-6, 1e-5, 1e-4]:
    for warmup_steps in [1000, 5000]:
      model, trainer = train(1, lr, warmup_steps)
      acc = evaluate_accuracy_torch(model, val_loader, device)
      print(lr, warmup_steps, acc)
      if acc['word_acc'] > max_word_acc:
        max_word_acc = acc['word_acc']
        trainer.save_model('torch_models/best_model')
        with open('best.txt', 'w') as fw:
          fw.write('Learning Rate: ' + str(lr) + '\n')
          fw.write('Warmup Steps: ' + str(warmup_steps) + '\n')
          fw.write(str(acc))

if __name__ == '__main__':
  model = DistilBertForTokenClassification.from_pretrained('torch_models/n41k')
  test_dataset = ViCharDiacDataset('data/vi/src_test.txt', 'data/vi/target_test.txt')
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
  print(evaluate_accuracy_torch(model, test_loader, device))
