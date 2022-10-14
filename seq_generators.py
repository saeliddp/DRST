import re
from constants import *
from viet_diacritic_mark import VietDiacriticMark, get_mark_viet

# training sequences of characters with labels being diacritics
def char_diac_seq_generator(src_fpath, target_fpath):
  src = open(src_fpath, 'r', encoding='utf-8')
  target = open(target_fpath, 'r', encoding='utf-8')
  pattern = r'[\s\W]'

  for src_line in src:
    target_line = target.readline()
    src_sylls = re.split(pattern, src_line)
    target_sylls = re.split(pattern, target_line)

    chars = []
    tags = []
    for src_syll, target_syll in zip(src_sylls, target_sylls):
      if src_syll and target_syll:
        chars.append(WORD_START)
        tags.append(VietDiacriticMark.NONE_NONE)
        if re.match(r'[0-9]', src_syll):
          chars.append(NUMERIC)
          tags.append(VietDiacriticMark.NONE_NONE)
        else:
          i = 0
          for src_char in src_syll:
            chars.append(src_char)
            tags.append(get_mark_viet(target_syll[i]))
            i += 1

    yield chars, tags

  src.close()
  target.close()

# training sequences of characters with labels being characters
def char_char_seq_generator(src_fpath, target_fpath):
  src = open(src_fpath, 'r', encoding='utf-8')
  target = open(target_fpath, 'r', encoding='utf-8')
  pattern = r'[\s\W]'

  for src_line in src:
    target_line = target.readline()
    src_sylls = re.split(pattern, src_line)
    target_sylls = re.split(pattern, target_line)

    chars = []
    tags = []
    for src_syll, target_syll in zip(src_sylls, target_sylls):
      if src_syll and target_syll:
        chars.append(WORD_START)
        tags.append(WORD_START)
        if re.match(r'[0-9]', src_syll):
          chars.append(NUMERIC)
          tags.append(NUMERIC)
        else:
          i = 0
          for src_char in src_syll:
            chars.append(src_char)
            tags.append(target_syll[i])
            i += 1

    yield chars, tags

  src.close()
  target.close()

# training sequences of syllables with labels being syllables
def syll_syll_seq_generator(src_fpath, target_fpath):
  src = open(src_fpath, 'r', encoding='utf-8')
  target = open(target_fpath, 'r', encoding='utf-8')
  pattern = r'[\s\W]'

  for src_line in src:
    target_line = target.readline()
    src_sylls = re.split(pattern, src_line)
    target_sylls = re.split(pattern, target_line)

    src_sylls_final = [SENT_START]
    target_sylls_final = [SENT_START]
    for src_syll, target_syll in zip(src_sylls, target_sylls):
      if src_syll and target_syll:
        if re.match(r'[0-9]', src_syll):
          src_sylls_final.append(NUMERIC)
          target_sylls_final.append(NUMERIC)
        else:
          src_sylls_final.append(src_syll)
          target_sylls_final.append(target_syll)

    yield src_sylls_final, target_sylls_final

  src.close()
  target.close()