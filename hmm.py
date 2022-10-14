import numpy as np
from utils import assign_ix_to_data
from constants import UNKNOWN

class HMM():
  # start_label is the label which will be at the front of every ground truth label sequence
  # e.g. START_SENT
  def __init__(self, start_label):
    self.start_label = start_label
    self.emission_probs = None # p(label | token)
    self.transition_probs = None # p(label | prev_label)
    self.sane_labels = {} # token -> set of observed labels (this only works for drst task, not general HMM)
    self.tok_to_ix = {}
    self.ix_to_tok = {}
    self.label_to_ix = {}
    self.ix_to_label = {}
  
  def train(self, tr_src_fpath, tr_target_fpath, generator_fn, smoothing=1, last_k_to_unknown=100):
    self.tok_to_ix, self.ix_to_tok, self.label_to_ix, self.ix_to_label = assign_ix_to_data(
      tr_src_fpath, tr_target_fpath, generator_fn, last_k_to_unknown=last_k_to_unknown)

    emission_counts = np.full((len(self.tok_to_ix), len(self.label_to_ix)), smoothing)
    transition_counts = np.full((len(self.label_to_ix), len(self.label_to_ix)), smoothing)

    for tokens, labels in generator_fn(tr_src_fpath, tr_target_fpath):
      prev_label = None
      for t, l in zip(tokens, labels):
        if t not in self.sane_labels:
          self.sane_labels[t] = set()
        self.sane_labels[t].add(l)

        if prev_label:
          transition_counts[self.label_to_ix[prev_label]][self.label_to_ix[l]] += 1
        
        if t in self.tok_to_ix:
          tok_ix = self.tok_to_ix[t]
        else:
          tok_ix = self.tok_to_ix[UNKNOWN]

        emission_counts[tok_ix][self.label_to_ix[l]] += 1
        prev_label = l
    
    if UNKNOWN in self.sane_labels:
      for label in label_to_ix:
        sane_labels[UNKNOWN].add(label)

    # move to log space
    emission_sums = np.log(np.sum(emission_counts, axis=1))
    self.emission_probs = np.log(emission_counts) - emission_sums.reshape(-1, 1)

    transition_sums = np.log(np.sum(transition_counts, axis=1))
    self.transition_probs = np.log(transition_counts) - transition_sums.reshape(-1, 1)
  
  def _viterbi_step(self, prev_scores, prev_label_indices, curr_token):
    scores = []
    label_indices = []
    bptrs = []

    for label in self.sane_labels[curr_token]:
      max_ind_val = (0, -np.inf)
      for i, prev_val in enumerate(prev_scores):
        prev_label_ind = prev_label_indices[i]
        val = prev_val + self.transition_probs[prev_label_ind][self.label_to_ix[label]]
        if curr_token in self.tok_to_ix:
          val += self.emission_probs[self.tok_to_ix[curr_token]][self.label_to_ix[label]]
        
        if val > max_ind_val[1]:
          max_ind_val = (i, val)

      scores.append(max_ind_val[1])
      label_indices.append(self.label_to_ix[label])
      bptrs.append(max_ind_val[0])
    
    return scores, label_indices, bptrs

  def _viterbi_predict(self, tokens):
    all_bptrs = []
    all_label_indices = []
    prev_scores = [0]
    prev_label_indices = [self.label_to_ix[self.start_label]]
    all_label_indices.append(prev_label_indices)

    for token in tokens[1: ]:
      scores, label_indices, bptrs = self._viterbi_step(prev_scores, prev_label_indices, token)
      all_bptrs.append(bptrs)
      all_label_indices.append(label_indices)
      prev_scores = scores
      prev_label_indices = label_indices
    
    best_end = np.argmax(prev_scores)
    label_indices = all_label_indices.pop()
    output = [self.ix_to_label[label_indices[best_end]]]

    while len(all_bptrs) > 0:
      bptrs = all_bptrs.pop()
      best_end = bptrs[best_end]
      label_indices = all_label_indices.pop()
      output.append(self.ix_to_label[label_indices[best_end]])
    
    output.reverse()
    return output

  def predict(self, tokens):
    return self._viterbi_predict(tokens)

if __name__ == '__main__':
  from seq_generators import *
  from evaluation import evaluate_accuracy
  import csv

  log_lines = []

  hmm_word = HMM(SENT_START)
  hmm_word.train('data/vi/src_train.txt', 'data/vi/target_train.txt', syll_syll_seq_generator)
  print('train done')
  acc = evaluate_accuracy(
    hmm_word, 'data/vi/src_test.txt', 'data/vi/target_test.txt', syll_syll_seq_generator, print_progress=True)
  log_lines.append(['syll_syll', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  print(acc)

  hmm_char = HMM(WORD_START)
  hmm_char.train('data/vi/src_train.txt', 'data/vi/target_train.txt', char_char_seq_generator)
  print('train done')
  acc = evaluate_accuracy(
    hmm_char, 'data/vi/src_test.txt', 'data/vi/target_test.txt', char_char_seq_generator, print_progress=True)
  log_lines.append(['char_char', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  print(acc)

  hmm_diac = HMM(WORD_START)
  hmm_diac.train('data/vi/src_train.txt', 'data/vi/target_train.txt', char_diac_seq_generator)
  print('train done')
  acc = evaluate_accuracy(
    hmm_diac, 'data/vi/src_test.txt', 'data/vi/target_test.txt', char_diac_seq_generator, print_progress=True)
  log_lines.append(['char_diac', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  print(acc)
  
  with open('test_results/hmm_accuracy.csv', 'w', newline='') as fw:
    writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sequence_label_types', 'character_accuracy', 'word_accuracy', 'sentence_accuracy'])
    for line in log_lines:
      writer.writerow(line)

