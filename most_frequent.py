import numpy as np
from utils import assign_ix_to_data

class MostFrequent():
  def __init__(self):
    self.most_freq_labels = None
    self.tok_to_ix = {}
    self.ix_to_tok = {}
    self.label_to_ix = {}
    self.ix_to_label = {}

  def train(self, tr_src_fpath, tr_target_fpath, generator_fn):
    self.tok_to_ix, self.ix_to_tok, self.label_to_ix, self.ix_to_label = assign_ix_to_data(tr_src_fpath, tr_target_fpath, generator_fn)
    
    counts = np.zeros((len(self.tok_to_ix), len(self.label_to_ix)))

    for tokens, labels in generator_fn(tr_src_fpath, tr_target_fpath):
      for t, l in zip(tokens, labels):
        counts[self.tok_to_ix[t]][self.label_to_ix[l]] += 1
    
    self.most_freq_labels = np.argmax(counts, axis=1)
  
  def predict(self, tokens):
    preds = []
    for t in tokens:
      if t in self.tok_to_ix:
        preds.append(self.ix_to_label[self.most_freq_labels[self.tok_to_ix[t]]])
      else:
        preds.append(t)
    return preds

if __name__ == '__main__':
  from seq_generators import *
  from evaluation import evaluate_accuracy
  import csv

  log_lines = []

  mf_word = MostFrequent()
  mf_word.train('data/vi/src_train.txt', 'data/vi/target_train.txt', syll_syll_seq_generator)
  acc = evaluate_accuracy(mf_word, 'data/vi/src_test.txt', 'data/vi/target_test.txt', syll_syll_seq_generator)
  log_lines.append(['syll_syll', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])

  mf_char = MostFrequent()
  mf_char.train('data/vi/src_train.txt', 'data/vi/target_train.txt', char_char_seq_generator)
  acc = evaluate_accuracy(mf_char, 'data/vi/src_test.txt', 'data/vi/target_test.txt', char_char_seq_generator)
  log_lines.append(['char_char', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])

  mf_diac = MostFrequent()
  mf_diac.train('data/vi/src_train.txt', 'data/vi/target_train.txt', char_diac_seq_generator)
  acc = evaluate_accuracy(mf_diac, 'data/vi/src_test.txt', 'data/vi/target_test.txt', char_diac_seq_generator)
  log_lines.append(['char_diac', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  
  with open('test_results/most_frequent_accuracy.csv', 'w', newline='') as fw:
    writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sequence_label_types', 'character_accuracy', 'word_accuracy', 'sentence_accuracy'])
    for line in log_lines:
      writer.writerow(line)