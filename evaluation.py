from utils import char_seq_to_syll_seq, diac_seq_to_syll_seq
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
    total_sentences += 1

  return {
    'char_acc': correct_chars / total_chars, 
    'word_acc': correct_words / total_words,
    'sentence_acc': correct_sentences / total_sentences
  }


