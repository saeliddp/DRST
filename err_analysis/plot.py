import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
  df = pd.read_csv('sent_freq.csv')
  plt.xlabel('relative frequency bucket')
  plt.ylabel('% of sentences containing >=1 word in bucket')
  plt.scatter(df['rel_freq'], df['percent_sent_containing'])
  plt.savefig('sent_perc.png')