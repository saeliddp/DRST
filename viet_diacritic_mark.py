from enum import Enum
from constants import WORD_START, NUMERIC, UNKNOWN, PAD

class VietDiacriticMark(Enum):
  # NONTONEMARK_TONEMARK
  NONE_NONE = 0
  NONE_GRAVE = 1
  NONE_ACUTE = 2
  NONE_HOOK = 3
  NONE_TILDE = 4
  NONE_DOT = 5

  CROSS_NONE = 6 # only used on d, which can't have tone mark

  HAT_NONE = 7
  HAT_GRAVE = 8
  HAT_ACUTE = 9
  HAT_HOOK = 10
  HAT_TILDE = 11
  HAT_DOT = 12

  TAIL_NONE = 13
  TAIL_GRAVE = 14
  TAIL_ACUTE = 15
  TAIL_HOOK = 16
  TAIL_TILDE = 17
  TAIL_DOT = 18

  SWOOP_NONE = 19
  SWOOP_GRAVE = 20
  SWOOP_ACUTE = 21
  SWOOP_HOOK = 22
  SWOOP_TILDE = 23
  SWOOP_DOT = 24

viet_char_to_mark = {} # will contain all besides NONE_NONE

for c in ['à', 'è', 'ì', 'ò', 'ù']: viet_char_to_mark[c] = VietDiacriticMark.NONE_GRAVE
for c in ['á', 'é', 'í', 'ó', 'ú']: viet_char_to_mark[c] = VietDiacriticMark.NONE_ACUTE
for c in ['ả', 'ẻ', 'ỉ', 'ỏ', 'ủ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_HOOK
for c in ['ã', 'ẽ', 'ĩ', 'õ', 'ũ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_TILDE
for c in ['ạ', 'ẹ', 'ị', 'ị', 'ụ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_DOT

viet_char_to_mark['đ'] = VietDiacriticMark.CROSS_NONE

for c in ['â', 'ê', 'ô']: viet_char_to_mark[c] = VietDiacriticMark.HAT_NONE
for c in ['ầ', 'ề', 'ồ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_GRAVE
for c in ['ấ', 'ế', 'ố']: viet_char_to_mark[c] = VietDiacriticMark.HAT_ACUTE
for c in ['ẩ', 'ể', 'ổ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_HOOK
for c in ['ẫ', 'ễ', 'ỗ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_TILDE
for c in ['ậ', 'ệ', 'ộ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_DOT

for c in ['ư', 'ơ']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_NONE
for c in ['ừ', 'ờ']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_GRAVE
for c in ['ứ', 'ớ']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_ACUTE
for c in ['ử', 'ở']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_HOOK
for c in ['ữ', 'ỡ']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_TILDE
for c in ['ự', 'ợ']: viet_char_to_mark[c] = VietDiacriticMark.TAIL_DOT

viet_char_to_mark['ă'] = VietDiacriticMark.SWOOP_NONE
viet_char_to_mark['ằ'] = VietDiacriticMark.SWOOP_GRAVE
viet_char_to_mark['ắ'] = VietDiacriticMark.SWOOP_ACUTE
viet_char_to_mark['ẳ'] = VietDiacriticMark.SWOOP_HOOK
viet_char_to_mark['ẵ'] = VietDiacriticMark.SWOOP_TILDE
viet_char_to_mark['ặ'] = VietDiacriticMark.SWOOP_DOT

def get_mark_viet(character):
  c = character.lower()
  return viet_char_to_mark[c] if c in viet_char_to_mark else VietDiacriticMark.NONE_NONE

# generated programmatically using diacritization_stripping_data and viet_char_to_mark
c_d_to_char = {('a', VietDiacriticMark.NONE_GRAVE): 'à', ('a', VietDiacriticMark.NONE_ACUTE): 'á', ('a', VietDiacriticMark.HAT_NONE): 'â', ('a', VietDiacriticMark.NONE_TILDE): 'ã', ('e', VietDiacriticMark.NONE_GRAVE): 'è', ('e', VietDiacriticMark.NONE_ACUTE): 'é', ('e', VietDiacriticMark.HAT_NONE): 'ê', ('i', VietDiacriticMark.NONE_GRAVE): 'ì', ('i', VietDiacriticMark.NONE_ACUTE): 'í', ('o', VietDiacriticMark.NONE_GRAVE): 'ò', ('o', VietDiacriticMark.NONE_ACUTE): 'ó', ('o', VietDiacriticMark.HAT_NONE): 'ô', ('o', VietDiacriticMark.NONE_TILDE): 'õ', ('u', VietDiacriticMark.NONE_GRAVE): 'ù', ('u', VietDiacriticMark.NONE_ACUTE): 'ú', ('a', VietDiacriticMark.SWOOP_NONE): 'ă', ('d', VietDiacriticMark.CROSS_NONE): 'đ', ('i', VietDiacriticMark.NONE_TILDE): 'ĩ', ('u', VietDiacriticMark.NONE_TILDE): 'ũ', ('o', VietDiacriticMark.TAIL_NONE): 'ơ', ('u', VietDiacriticMark.TAIL_NONE): 'ư', ('a', VietDiacriticMark.NONE_DOT): 'ạ', ('a', VietDiacriticMark.NONE_HOOK): 'ả', ('a', VietDiacriticMark.HAT_ACUTE): 'ấ', ('a', VietDiacriticMark.HAT_GRAVE): 'ầ', ('a', VietDiacriticMark.HAT_HOOK): 'ẩ', ('a', VietDiacriticMark.HAT_TILDE): 'ẫ', ('a', VietDiacriticMark.HAT_DOT): 'ậ', ('a', VietDiacriticMark.SWOOP_ACUTE): 'ắ', ('a', VietDiacriticMark.SWOOP_GRAVE): 'ằ', ('a', VietDiacriticMark.SWOOP_HOOK): 'ẳ', ('a', VietDiacriticMark.SWOOP_TILDE): 'ẵ', ('a', VietDiacriticMark.SWOOP_DOT): 'ặ', ('e', VietDiacriticMark.NONE_DOT): 'ẹ', ('e', VietDiacriticMark.NONE_HOOK): 'ẻ', ('e', VietDiacriticMark.NONE_TILDE): 'ẽ', ('e', VietDiacriticMark.HAT_ACUTE): 'ế', ('e', VietDiacriticMark.HAT_GRAVE): 'ề', ('e', VietDiacriticMark.HAT_HOOK): 'ể', ('e', VietDiacriticMark.HAT_TILDE): 'ễ', ('e', VietDiacriticMark.HAT_DOT): 'ệ', ('i', VietDiacriticMark.NONE_HOOK): 'ỉ', ('i', VietDiacriticMark.NONE_DOT): 'ị', ('o', VietDiacriticMark.NONE_HOOK): 'ỏ', ('o', VietDiacriticMark.HAT_ACUTE): 'ố', ('o', VietDiacriticMark.HAT_GRAVE): 'ồ', ('o', VietDiacriticMark.HAT_HOOK): 'ổ', ('o', VietDiacriticMark.HAT_TILDE): 'ỗ', ('o', VietDiacriticMark.HAT_DOT): 'ộ', ('o', VietDiacriticMark.TAIL_ACUTE): 'ớ', ('o', VietDiacriticMark.TAIL_GRAVE): 'ờ', ('o', VietDiacriticMark.TAIL_HOOK): 'ở', ('o', VietDiacriticMark.TAIL_TILDE): 'ỡ', ('o', VietDiacriticMark.TAIL_DOT): 'ợ', ('u', VietDiacriticMark.NONE_DOT): 'ụ', ('u', VietDiacriticMark.NONE_HOOK): 'ủ', ('u', VietDiacriticMark.TAIL_ACUTE): 'ứ', ('u', VietDiacriticMark.TAIL_GRAVE): 'ừ', ('u', VietDiacriticMark.TAIL_HOOK): 'ử', ('u', VietDiacriticMark.TAIL_TILDE): 'ữ', ('u', VietDiacriticMark.TAIL_DOT): 'ự'}

def rediacritize_viet_char(char, mark):
  return c_d_to_char[(char, mark)] if (char, mark) in c_d_to_char else char if mark == VietDiacriticMark.NONE_NONE else char + '*'

# valid vietnamese characters + special characters
src_char_to_ix = {c: i + 4 for i, c in enumerate('abcdeghiklmnopqrstuvxy')}
src_char_to_ix[WORD_START] = 0
src_char_to_ix[NUMERIC] = 1
src_char_to_ix[UNKNOWN] = 2
src_char_to_ix[PAD] = 3

label_to_ix = {data: data.value for data in VietDiacriticMark}

ix_to_src_char = {v: k for k, v in src_char_to_ix.items()}
ix_to_label = {v: k for k, v in label_to_ix.items()}