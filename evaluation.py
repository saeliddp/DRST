import csv
import pandas as pd
import math
import numpy as np
from utils import char_seq_to_syll_seq, diac_seq_to_syll_seq
from constants import NUMERIC
import seq_generators

def evaluate_accuracy(model, test_src_fpath, test_target_fpath, generator_fn, print_progress=False):
  total_chars = 0
  correct_chars = 0
  total_words = 0
  correct_words = 0
  total_sentences = 0
  correct_sentences = 0

  for tokens, labels in generator_fn(test_src_fpath, test_target_fpath):
    if print_progress:
      print(total_sentences, end='\r')
      
    preds = model.predict(tokens)

    # convert all predictions to word predictions
    if generator_fn == seq_generators.syll_syll_seq_generator:
      true_words = labels
      pred_words = preds
    elif generator_fn == seq_generators.char_char_seq_generator:
      true_words = char_seq_to_syll_seq(labels)
      pred_words = char_seq_to_syll_seq(preds)
    else: # implicitly char_diac_seq_generator
      true_words = diac_seq_to_syll_seq(tokens, labels)
      pred_words = diac_seq_to_syll_seq(tokens, preds)
    
    sentence_corr = True
    for t_w, p_w in zip(true_words[1:], pred_words[1:]): # skip START_SENT
      if t_w == NUMERIC:
        continue

      word_corr = True
      # technically possible that lengths are different for word-based models
      for i in range(min(len(t_w), len(p_w))):
        if t_w[i] == p_w[i]:
          correct_chars += 1
        else:
          word_corr = False
      # penalizes equally if p_w is shorter or longer than t_w
      total_chars += max(len(t_w), len(p_w))
      if len(t_w) != len(p_w):
        word_corr = False
      
      if word_corr:
        correct_words += 1
      else:
        sentence_corr = False
      total_words += 1
    
    if sentence_corr:
      correct_sentences += 1
    else:
      print(true_words)
      print(pred_words)
      print()
    total_sentences += 1

  return {
    'char_acc': correct_chars / total_chars, 
    'word_acc': correct_words / total_words,
    'sentence_acc': correct_sentences / total_sentences
  }

def evaluate_accuracy_per_src_syllable(model, test_src_fpath, test_target_fpath, generator_fn, 
                                   print_progress=False, save_to_csv=None):
  token_counts = {}

  total_sentences = 0
  for tokens, labels in generator_fn(test_src_fpath, test_target_fpath):
    if print_progress:
      print(total_sentences, end='\r')
      
    preds = model.predict(tokens)

    # convert all predictions to word predictions
    if generator_fn == seq_generators.syll_syll_seq_generator:
      src_words = tokens
      true_words = labels
      pred_words = preds
    elif generator_fn == seq_generators.char_char_seq_generator:
      src_words = char_seq_to_syll_seq(tokens)
      true_words = char_seq_to_syll_seq(labels)
      pred_words = char_seq_to_syll_seq(preds)
    else: # implicitly char_diac_seq_generator
      src_words = char_seq_to_syll_seq(tokens)
      true_words = diac_seq_to_syll_seq(tokens, labels)
      pred_words = diac_seq_to_syll_seq(tokens, preds)
    
    for s_w, t_w, p_w in zip(src_words[1:], true_words[1:], pred_words[1:]): # skip START_SENT
      if t_w == NUMERIC:
        continue
      if s_w not in token_counts:
        token_counts[s_w] = {}
        token_counts[s_w]['total'] = 0
        token_counts[s_w]['correct'] = 0
      token_counts[s_w]['total'] += 1
      if t_w == p_w:
        token_counts[s_w]['correct'] += 1
    total_sentences += 1

  accuracies = {}
  for k in token_counts:
    accuracies[k] = token_counts[k]['correct'] / token_counts[k]['total']
  
  if save_to_csv:
    with open(save_to_csv, 'w', newline='') as fw:
      writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['src_token', 'accuracy'])
      for k in accuracies:
        writer.writerow([k, accuracies[k]])

  return accuracies

def evaluate_prf_per_target_syllable(model, test_src_fpath, test_target_fpath, generator_fn, 
                                   print_progress=False, save_to_csv=None):
  token_counts = {}

  total_sentences = 0
  for tokens, labels in generator_fn(test_src_fpath, test_target_fpath):
    if print_progress:
      print(total_sentences, end='\r')
      
    preds = model.predict(tokens)

    # convert all predictions to word predictions
    if generator_fn == seq_generators.syll_syll_seq_generator:
      src_words = tokens
      true_words = labels
      pred_words = preds
    elif generator_fn == seq_generators.char_char_seq_generator:
      src_words = char_seq_to_syll_seq(tokens)
      true_words = char_seq_to_syll_seq(labels)
      pred_words = char_seq_to_syll_seq(preds)
    else: # implicitly char_diac_seq_generator
      src_words = char_seq_to_syll_seq(tokens)
      true_words = diac_seq_to_syll_seq(tokens, labels)
      pred_words = diac_seq_to_syll_seq(tokens, preds)
    
    for s_w, t_w, p_w in zip(src_words[1:], true_words[1:], pred_words[1:]): # skip START_SENT
      if t_w == NUMERIC:
        continue
      if t_w not in token_counts:
        token_counts[t_w] = {}
        token_counts[t_w]['times_predicted'] = 0
        token_counts[t_w]['times_correct'] = 0
        token_counts[t_w]['correct'] = 0
      token_counts[t_w]['times_correct'] += 1

      if p_w not in token_counts:
        token_counts[p_w] = {}
        token_counts[p_w]['times_predicted'] = 0
        token_counts[p_w]['times_correct'] = 0
        token_counts[p_w]['correct'] = 0
      token_counts[p_w]['times_predicted'] += 1

      if t_w == p_w:
        token_counts[t_w]['correct'] += 1
    total_sentences += 1

  recalls = {}
  precisions = {}
  f_ones = {}
  for k in token_counts:
    if token_counts[k]['times_correct'] > 0:
      recalls[k] = token_counts[k]['correct'] / token_counts[k]['times_correct']
    else:
      recalls[k] = -1

    if token_counts[k]['times_predicted'] > 0:
      precisions[k] = token_counts[k]['correct'] / token_counts[k]['times_predicted']
    else:
      precisions[k] = -1
      
    if recalls[k] > 0 and precisions[k] > 0:
      f_ones[k] = 2 * (precisions[k] * recalls[k]) / (precisions[k] + recalls[k])
    else:
      f_ones[k] = -1
  
  if save_to_csv:
    with open(save_to_csv, 'w', newline='') as fw:
      writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['target_token', 'recall', 'precision', 'f1'])
      for k in recalls:
        writer.writerow([k, recalls[k], precisions[k], f_ones[k]])

  return recalls, precisions, f_ones

def evaluate_sent_accuracy_by_rf(model, test_src_fpath, test_target_fpath, generator_fn, 
                                 bucket_size=0.025, save_to_csv=None):
  tgt_syll_rfs = pd.read_csv('err_analysis/tgt_syll_rel_freq.csv', na_values='NaN', keep_default_na=False)
  rfs = {}
  for row in tgt_syll_rfs.itertuples():
    rfs[row.target_token]= row.rel_freq

  preds_by_rf = {}
  for c in range(0, int(1 / bucket_size) + 1):
    rf = c / (1 / bucket_size)
    preds_by_rf[rf] = {'total': 0, 'correct': 0}
  preds_by_rf[-1] = {'total': 0, 'correct': 0} # unknown word rf
  
  num_sentences = 0
  for tokens, labels in generator_fn(test_src_fpath, test_target_fpath):
    num_sentences += 1
    preds = model.predict(tokens)

    # convert all predictions to word predictions
    if generator_fn == seq_generators.syll_syll_seq_generator:
      true_words = labels
      pred_words = preds
    elif generator_fn == seq_generators.char_char_seq_generator:
      true_words = char_seq_to_syll_seq(labels)
      pred_words = char_seq_to_syll_seq(preds)
    else: # implicitly char_diac_seq_generator
      true_words = diac_seq_to_syll_seq(tokens, labels)
      pred_words = diac_seq_to_syll_seq(tokens, preds)
    
    sentence_corr = True
    rfs_for_sent = set()
    for t_w, p_w in zip(true_words[1:], pred_words[1:]): # skip START_SENT
      if t_w == NUMERIC:
        continue

      if t_w not in rfs:
        rfs_for_sent.add(-1)
      else:
        div = 1 / bucket_size
        val = math.floor(rfs[t_w] / bucket_size)
        rfs_for_sent.add(val / div)
      
      if t_w != p_w:
        sentence_corr = False
    
    for rf in rfs_for_sent:
      preds_by_rf[rf]['total'] += 1
      if sentence_corr:
        preds_by_rf[rf]['correct'] += 1

  if save_to_csv:
    """with open(save_to_csv, 'w', newline='') as fw:
      writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['rel_freq', 'sentence_acc'])
      for rf in preds_by_rf:
        if preds_by_rf[rf]['total'] > 0:
          writer.writerow([rf, preds_by_rf[rf]['correct'] / preds_by_rf[rf]['total']])"""
    with open('err_analysis/sent_freq.csv', 'w', newline='') as fw:
      writer = csv.writer(fw, quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['rel_freq', 'percent_sent_containing'])
      for rf in preds_by_rf:
        writer.writerow([rf, preds_by_rf[rf]['total'] / num_sentences])
