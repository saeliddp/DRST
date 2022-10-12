from constants import *
import re

def char_seq_generator(src_fpath: str, target_fpath: str):
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