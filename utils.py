from collections import Counter
from constants import *
from viet_diacritic_mark import rediacritize_viet_char

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