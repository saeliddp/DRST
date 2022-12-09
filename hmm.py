import numpy as np
from utils import assign_ix_to_data
from constants import *

class HMM():
  # start_label is the label which will be at the front of every ground truth label sequence
  # e.g. START_SENT
  def __init__(self, start_label):
    self.start_label = start_label
    self.emission_probs = None # p(token | label)
    self.transition_probs = None # p(label | prev_label)
    self.poss_labels = None
    self.tok_to_ix = {}
    self.ix_to_tok = {}
    self.label_to_ix = {}
    self.ix_to_label = {}

  def _smooth_convert_count_to_prob(self, key, count_dict, smoothing):
    # all emission count keys will not be empty, so we won't run into division by 0 with
    # smoothing=0 for emission; the same cannot be said for transition counts (a word could
    # only ever show up at end of sentence) so we need smoothing
    sum_count = np.sum([count_dict[key][k] for k in count_dict[key]])
    sum_count += len(self.label_to_ix) * smoothing
    sum_count = np.log(sum_count)
    for k in count_dict[key]:
      count_dict[key][k] = np.log(count_dict[key][k] + smoothing) - sum_count
    
    if smoothing == 0:
      count_dict[key][OTHER] = -np.inf
    else:
      count_dict[key][OTHER] = np.log(smoothing) - sum_count
  
  def train(self, tr_src_fpath, tr_target_fpath, generator_fn, smoothing=1):
    self.tok_to_ix, self.ix_to_tok, self.label_to_ix, self.ix_to_label = assign_ix_to_data(
      tr_src_fpath, tr_target_fpath, generator_fn)

    emission_counts = {label: {} for label in self.label_to_ix}
    transition_counts = {label: {} for label in self.label_to_ix}
    
    self.poss_labels = {token: set() for token in self.tok_to_ix}
    transition_counts[UNKNOWN] = {}

    for tokens, labels in generator_fn(tr_src_fpath, tr_target_fpath):
      prev_label = None
      for t, l in zip(tokens, labels):

        if prev_label:
          if l not in transition_counts[prev_label]:
            transition_counts[prev_label][l] = 0

          transition_counts[prev_label][l] += 1

        self.poss_labels[t].add(l)
        if t not in emission_counts[l]:
          emission_counts[l][t] = 0
        
        emission_counts[l][t] += 1
        prev_label = l

    for token in emission_counts:
      self._smooth_convert_count_to_prob(token, emission_counts, 0)
    self.emission_probs = emission_counts 

    for label in transition_counts:
      self._smooth_convert_count_to_prob(label, transition_counts, smoothing)
    self.transition_probs = transition_counts
  
  def _viterbi_step(self, prev_scores, prev_labels, curr_token):
    scores = []
    labels = []
    bptrs = []

    # if we haven't seen curr_token in training data, then we have no
    # sane/feasible tags we could assign it, so we assign it to itself
    if curr_token not in self.tok_to_ix:
      max_ind = np.argmax(prev_scores)
      scores.append(prev_scores[max_ind])
      labels.append(curr_token)
      bptrs.append(max_ind)
      return scores, labels, bptrs

    for label in self.poss_labels[curr_token]:
      # safe to ignore since emission prob will be 0 and we know there will be another label
      # besides OTHER
      if label == OTHER:
        continue

      max_ind_val = (0, -np.inf)
      for i, prev_val in enumerate(prev_scores):
        prev_label = prev_labels[i]
        if prev_label in self.transition_probs:
          if label in self.transition_probs[prev_label]:
            val = prev_val + self.transition_probs[prev_label][label]
          else:
            val = prev_val + self.transition_probs[prev_label][OTHER]
        else:
          val = prev_val + self.transition_probs[UNKNOWN][OTHER]

        val += self.emission_probs[label][curr_token]
        
        if val > max_ind_val[1]:
          max_ind_val = (i, val)

      scores.append(max_ind_val[1])
      labels.append(label)
      bptrs.append(max_ind_val[0])
    
    return scores, labels, bptrs

  def _viterbi_predict(self, tokens):
    all_bptrs = []
    all_labels = []
    prev_scores = [0]
    prev_labels = [self.start_label]
    all_labels.append(prev_labels)

    for token in tokens[1: ]:
      scores, labels, bptrs = self._viterbi_step(prev_scores, prev_labels, token)
      all_bptrs.append(bptrs)
      all_labels.append(labels)
      prev_scores = scores
      prev_labels = labels
    
    best_end = np.argmax(prev_scores)
    labels = all_labels.pop()
    output = [labels[best_end]]

    while len(all_bptrs) > 0:
      bptrs = all_bptrs.pop()
      best_end = bptrs[best_end]
      labels = all_labels.pop()
      output.append(labels[best_end])
    
    output.reverse()
    return output

  def predict(self, tokens):
    return self._viterbi_predict(tokens)

if __name__ == '__main__':
  from seq_generators import *
  from evaluation import *
  from viet_diacritic_mark import VietDiacriticMark
  import csv

  log_lines = []
  
  hmm_char = HMM(SENT_START)
  hmm_char.train('data/vi/src_train.txt', 'data/vi/target_train.txt', syll_syll_seq_generator)
  print('train done')
  acc = evaluate_accuracy(
    hmm_char, 'data/vi/src_test.txt', 'data/vi/target_test.txt', syll_syll_seq_generator, print_progress=True)
  log_lines.append(['syll_syll', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  print(acc)

  hmm_char = HMM(WORD_START)
  hmm_char.train('data/vi/src_train.txt', 'data/vi/target_train.txt', char_char_seq_generator)
  print('train done')
  acc = evaluate_accuracy(
    hmm_char, 'data/vi/src_test.txt', 'data/vi/target_test.txt', char_char_seq_generator, print_progress=True)
  log_lines.append(['char_char', acc['char_acc'], acc['word_acc'], acc['sentence_acc']])
  print(acc)

  hmm_diac = HMM(VietDiacriticMark.NONE_NONE)
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
  
