from enum import Enum

class VietDiacriticMark(Enum):
  # NONTONEMARK_TONEMARK
  NONE_NONE = 0
  NONE_GRAVE = 1
  NONE_ACUTE = 2
  NONE_HOOK = 3
  NONE_TILDE = 4
  NONE_DOT = 5

  CROSS_NONE = 6 # only used on d, which can't have accent

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

viet_char_to_mark = {}

for c in ['à', 'è', 'ì', 'ò', 'ù']: viet_char_to_mark[c] = VietDiacriticMark.NONE_GRAVE
for c in ['á', 'é', 'í', 'ó', 'ú']: viet_char_to_mark[c] = VietDiacriticMark.NONE_ACUTE
for c in ['ả', 'ẻ', 'ỉ', 'ỏ', 'ủ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_HOOK
for c in ['ã', 'ẽ', 'ĩ', 'õ', 'ũ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_TILDE
for c in ['ạ', 'ẹ', 'ị', 'ị', 'ụ']: viet_char_to_mark[c] = VietDiacriticMark.NONE_DOT

viet_char_to_mark['đ'] = VietDiacriticMark.CROSS_NONE

for c in ['à', 'è', 'ì', 'ò', 'ù']: viet_char_to_mark[c] = VietDiacriticMark.HAT_GRAVE
for c in ['á', 'é', 'í', 'ó', 'ú']: viet_char_to_mark[c] = VietDiacriticMark.HAT_ACUTE
for c in ['ả', 'ẻ', 'ỉ', 'ỏ', 'ủ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_HOOK
for c in ['ã', 'ẽ', 'ĩ', 'õ', 'ũ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_TILDE
for c in ['ạ', 'ẹ', 'ị', 'ị', 'ụ']: viet_char_to_mark[c] = VietDiacriticMark.HAT_DOT