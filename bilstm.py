import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from seq_generators import *
from constants import UNKNOWN
from evaluation import evaluate_accuracy
from utils import assign_ix_to_data

def prepare_sequence(seq, to_ix):
  idxs = [to_ix[w] if w in to_ix else to_ix[UNKNOWN] for w in seq]
  tensor = torch.LongTensor(idxs)
  return Variable(tensor)

def to_scalar(var):
  # returns a python float
  return var.view(-1).data.tolist()

def argmax(vec):
  # return the argmax as a python int
  _, idx = torch.max(vec, 1)
  return to_scalar(idx)

class BiLSTM(nn.Module):
  """
  Class for the BiLSTM model tagger
  """

  def __init__(self, token_to_ix, label_to_ix, ix_to_label, embedding_dim, hidden_dim, embeddings=None):
    super(BiLSTM, self).__init__()
    
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = len(token_to_ix)
    self.token_to_ix = token_to_ix
    self.label_to_ix = label_to_ix
    self.ix_to_label = ix_to_label
    self.tagset_size = len(label_to_ix)
    
    """
    name them as following:
    self.word_embeds: embedding variable
    self.lstm: lstm layer
    self.hidden2tag: fully connected layer
    """
    
    self.word_embeds = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
    
    if embeddings is not None:
      self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
    
    # Maps the embeddings of the word into the hidden state
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)

    # Maps the output of the LSTM into tag space
    self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=self.tagset_size, bias=True)
    self.hidden = self.init_hidden()

  def init_hidden(self):
    # axes semantics are: bidirectinal*num_of_layers, minibatch_size, hidden_dimension; we use noisy initialization
    
    return (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
            Variable(torch.randn(2, 1, self.hidden_dim // 2)))

  def forward(self, sentence):
    """
    The function obtain the scores for each tag for each of the words in a sentence
    Input:
    sentence: a sequence of ids for each word in the sentence
    Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
    The axes semantics are: seq_len, mini_batch, embedding_dim
    Output: 
    returns lstm_feats: scores for each tag for each token in the sentence.
    """
    self.hidden = self.init_hidden()
    sentence = prepare_sequence(sentence, self.token_to_ix)
    embeddings = self.word_embeds(sentence)
    embeddings = embeddings.view(len(sentence), 1, self.embedding_dim)
    lstm_out, (hn, cn) = self.lstm(embeddings, self.hidden)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    linear_out = self.hidden2tag(lstm_out)
    return linear_out
      
  def predict(self, sentence):
    """
    this function is used for evaluating the model: 
    Input:
        sentence: a sequence of ids for each word in the sentence
    Outputs:
        Obtains the scores for each token by passing through forward, then passes the scores for each token 
        through a softmax-layer and then predicts the tag with the maximum probability for each token: 
        observe that this is like greedy decoding
    """
    lstm_feats = self.forward(sentence)
    softmax_layer = torch.nn.Softmax(dim=1)
    probs = softmax_layer(lstm_feats)
    idx = argmax(probs)
    tags = [self.ix_to_label[ix] for ix in idx]
    return tags

def train_model(loss, model, src_tr, target_tr, token_to_ix, label_to_ix, seq_generator,
               src_dv, target_dv, num_its=50, status_frequency=1,
               optim_args = {'lr':0.1,'momentum':0},
               param_file = 'best.params'):
    
  #initialize optimizer
  optimizer = optim.SGD(model.parameters(), **optim_args)
  
  losses=[]
  accuracies=[]
  max_word_acc = 0
  for epoch in range(num_its):
    
    loss_value=0
    count1=0
    
    for X, Y in seq_generator(src_tr, target_tr):
      Y_tr_var = prepare_sequence(Y, label_to_ix)
      
      # set gradient to zero
      optimizer.zero_grad()
      
      if len(X) > 0:
        lstm_feats= model.forward(X)
        output = loss(lstm_feats,Y_tr_var)
        
        output.backward()
        optimizer.step()
        loss_value += output.item()
      count1+=1
      print('Sequence #', count1, end='\r')
        
    print()
    losses.append(loss_value/count1)
    
    accuracy = evaluate_accuracy(model, src_dv, target_dv, generator_fn)
    accuracies.append(accuracy)
    if accuracy['word_acc'] > max_word_acc:
      max_word_acc = accuracy['word_acc']
      state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':accuracy['word_acc']}
      torch.save(state,'bilstm_best.params')

    # print status message if desired
    if status_frequency > 0 and epoch % status_frequency == 0:
        print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(accuracy))

  return model, losses, accuracies

if __name__ == '__main__':
  tr_src = 'data/vi/src_train.txt'
  tr_target = 'data/vi/target_train.txt'
  dev_src = 'data/vi/src_dev.txt'
  dev_target = 'data/vi/target_dev.txt'
  generator_fn = syll_syll_seq_generator
  loss = torch.nn.CrossEntropyLoss()

  tok_to_ix, ix_to_tok, label_to_ix, ix_to_label = assign_ix_to_data(
      tr_src, tr_target, generator_fn, last_k_to_unknown=2)
  
  bilstm = BiLSTM(tok_to_ix, label_to_ix, ix_to_label, 30, 30)

  train_model(loss, bilstm, tr_src, tr_target, tok_to_ix, label_to_ix, generator_fn,
               dev_src, dev_target)

