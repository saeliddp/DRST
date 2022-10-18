# Experimental Setup
### Dataset
https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2607  
Using the Wikipedia portion of the Vietnamese data linked above.  
Train Size: 819,918 sentences  
Dev Size:   14,884 sentences  
Test Size:  30,000 sentences  

The data comes lightly preprocessed (1 sentence per line, all lowercased)
### Tokenization
To get syllables/words from raw sentences, split on r'[\s\W]'

Replace any syllables containing 0-9 with NUMERIC token ('#')

When treating sequences as syllable/word sequences, prepend SENT_START token ('@') as first token in all sentences. E.g. ['@', 'hom', 'nay', ...]

When treating sequences as char sequences, prepend each word's sequence of characters with WORD_START token ('^'). E.g. ['^', 'h', 'o', 'm', '^', 'n', 'a', 'y', ...]

### Sequence types
1. Word-word / syllable-syllable: observed sequences are asciified words, target tags are diacritized words
2. Char-char: observed sequences are asciified characters, target tags are diacritized characters
3. Char-diac: observed sequences are asciified characters, target tags are diacritic marks for those characters

### Models
- Most Frequent: assign each input token its most frequently observed tags. Unseen input tokens are kept as-is.
- HMM: choose most probable path through tags based on tag emission probabilities and transition probabilities. Unseen input tokens are kept as-is.