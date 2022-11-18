import csv
import torch
from collections import Counter
from constants import *
from viet_diacritic_mark import *

# if last_k_to_unknown > 0, coalesces least-k frequent tokens to UNKNOWN token
def assign_ix_to_data(tr_src_fpath, tr_target_fpath, generator_fn, last_k_to_unknown=0):
  tok_to_ix = {}
  ix_to_tok = {}
  label_to_ix = {}
  ix_to_label = {}

  curr_label_ix = 0
  token_counts = Counter()

  for tokens, labels in generator_fn(tr_src_fpath, tr_target_fpath):
    for token in tokens:
      token_counts[token] += 1
    
    for label in labels:
      if label not in label_to_ix:
        label_to_ix[label] = curr_label_ix
        ix_to_label[curr_label_ix] = label
        curr_label_ix += 1

  if last_k_to_unknown > 0:
    unk_count = 0
    for token, count in token_counts.most_common()[-last_k_to_unknown: ]:
      unk_count += count
      del token_counts[token]
    token_counts[UNKNOWN] = unk_count
  
  curr_token_ix = 0
  for token in token_counts:
    if token not in tok_to_ix:
        tok_to_ix[token] = curr_token_ix
        ix_to_tok[curr_token_ix] = token
        curr_token_ix += 1

  return tok_to_ix, ix_to_tok, label_to_ix, ix_to_label

def get_token_freq_branching_factor(src_fpaths, target_fpaths, generator_fn, save_to_csv=None):
    frequencies = Counter()
    label_possibilities = {}
    for src_fpath, target_fpath in zip(src_fpaths, target_fpaths):
      for tokens, labels in generator_fn(src_fpath, target_fpath):
        for t, l in zip(tokens, labels):
          if t in SPECIAL_SEP_CHARS:
            continue
          frequencies[t] += 1
          if t not in label_possibilities:
            label_possibilities[t] = set()
          label_possibilities[t].add(l)

    branching_factors = Counter()
    for k in label_possibilities:
      branching_factors[k] = len(label_possibilities[k])

    if save_to_csv:
      with open(save_to_csv, 'w', newline='') as fw:
        writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['src_token', 'frequency', 'branching_factor'])
        for k in frequencies:
          writer.writerow([k, frequencies[k], branching_factors[k]])
    
    return frequencies, branching_factors

def get_token_rel_freq(src_fpaths, target_fpaths, generator_fn, save_to_csv=None):
    frequencies = {}
    for src_fpath, target_fpath in zip(src_fpaths, target_fpaths):
      for tokens, labels in generator_fn(src_fpath, target_fpath):
        for t, l in zip(tokens, labels):
          if t in SPECIAL_SEP_CHARS:
            continue
          if t not in frequencies:
            frequencies[t] = Counter()
          frequencies[t][l] += 1

    rel_freqs = {}
    for k in frequencies:
      total = sum([frequencies[k][l] for l in frequencies[k]])
      for l in frequencies[k]:
        rel_freqs[l] = frequencies[k][l] / total

    if save_to_csv:
      with open(save_to_csv, 'w', newline='') as fw:
        writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['target_token', 'rel_freq'])
        for l in rel_freqs:
          writer.writerow([l, rel_freqs[l]])
    
    return rel_freqs

# given a sequence of WORD_START separated chars (e.g. ['^', 'h', 'i', '^', 'J', 'o'])
# return a syllable sequence (e.g. ['@', 'hi', 'Jo])
def char_seq_to_syll_seq(char_seq):
  syll_seq = []
  curr_end = len(char_seq)
  curr_start = len(char_seq) - 1
  while curr_start >= 0:
    if char_seq[curr_start] == WORD_START:
      syll_seq.append(''.join(char_seq[curr_start + 1: curr_end]))
      curr_end = curr_start
    curr_start -= 1
  syll_seq.append(SENT_START)
  syll_seq.reverse()
  return syll_seq

# given a sequence of WORD_START separated chars (e.g. ['^', 'h', 'i', '^', 'J', 'o'])
# and the corresponding proper VietDiacriticMark sequence, return diacritized syllables
def diac_seq_to_syll_seq(tokens, diacritics):
  syll_seq = []
  curr_end = len(tokens)
  curr_start = len(tokens) - 1

  curr_word = []
  while curr_start >= 0:
    if tokens[curr_start] == WORD_START:
      curr_word.reverse()
      syll_seq.append(''.join(curr_word))
      curr_end = curr_start
      curr_word = []
    else:
      curr_word.append(rediacritize_viet_char(tokens[curr_start], diacritics[curr_start]))
    curr_start -= 1
  syll_seq.append(SENT_START)
  syll_seq.reverse()
  return syll_seq

def torch_ind_to_syll_seq(token_ids, label_ids):
  tokens = [ix_to_src_char[int(t)] for t in token_ids]
  end_ix = tokens.index(PAD) if PAD in tokens else len(tokens)
  tokens = tokens[:end_ix]
  labels = [ix_to_label[int(l)] for l in label_ids[:end_ix]]
  return diac_seq_to_syll_seq(tokens, labels)

def reconstruct_torch_preds(dataset, model, limit=5):
  all_targets = []
  all_preds = []
  for i in range(limit):
    if i >= len(dataset.encodings):
      break
    ids = dataset.encodings[i].ids
    all_targets.append(torch_ind_to_syll_seq(ids, dataset.labels[i]))
    logits = model(torch.tensor([ids])).logits
    preds = [ix_to_label[l.item()] for l in torch.argmax(logits, axis=2).flatten()]
    all_preds.append(torch_ind_to_syll_seq(ids, preds))

  return all_targets, all_preds