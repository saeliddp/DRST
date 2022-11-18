import torch
import numpy as np
from constants import *
from seq_generators import char_diac_seq_generator
from viet_diacritic_mark import src_char_to_ix, label_to_ix, VietDiacriticMark
from tokenizers import (
  decoders,
  models,
  normalizers,
  pre_tokenizers,
  processors,
  trainers,
  Tokenizer,
)
from transformers import PreTrainedTokenizerFast

tokenizer_obj = Tokenizer(models.WordLevel(unk_token=UNKNOWN, vocab=src_char_to_ix))
tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj,
    unk_token=UNKNOWN,
    pad_token=PAD,
    model_max_length=300,
)

def encode_tags(tags, encodings):
  encoded_labels = []
  for doc_labels, input_ids in zip(tags, encodings.input_ids):
    enc_labels = []
    for i in range(len(input_ids)):
      if input_ids[i] == src_char_to_ix[PAD]:
        enc_labels.append(-100)
      else:
        label = doc_labels[i]
        enc_labels.append(label_to_ix[label])
    encoded_labels.append(enc_labels)

  return encoded_labels

class ViCharDiacDataset(torch.utils.data.Dataset):
  def __init__(self, src_fpath, target_fpath):
    all_tokens = []
    all_labels = []
    count = 0
    for tokens, labels in char_diac_seq_generator(src_fpath, target_fpath):
      if len(tokens) > tokenizer.model_max_length:
        count += 1
      all_tokens.append(tokens)
      all_labels.append(labels)
    #print(str(count) + '/' + str(len(all_tokens)))
    self.encodings = tokenizer(all_tokens, is_split_into_words=True, return_offsets_mapping=False, padding=True, truncation=True)
    self.labels = encode_tags(all_labels, self.encodings)

  def __getitem__(self, idx):
    item = {'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            #'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])}
    return item

  def __len__(self):
    return len(self.labels)