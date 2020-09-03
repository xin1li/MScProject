
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
import json
from random import sample


batch_size = 1000
Learning_Rate = 0.01

def json2dataframe(file_path):

    labels, input_sentences = [], [] 

    with open(file_path) as json_file:
        data = json.load(json_file)
        for index in range(len(data)):
            chosen_model = data[index]["chosen_suggestions"]
            for ug_speaker_ind in range(1,len(chosen_model),2):
                label = chosen_model[ug_speaker_ind]
                if label == '':
                    label = "manual"
                #print(label)
                g_speaker_input = data[index]["dialog"][ug_speaker_ind-1][1]
                #print(g_speaker_input)
                labels.append(label)
                input_sentences.append(g_speaker_input)

    dat = pd.DataFrame()
    dat['input sentence'] = input_sentences
    dat['label'] = labels

    print(dat.shape)
    return dat 

train = json2dataframe("/Users/xinyilihuang/ParlAI/data/blended_skill_talk/train.json")
valid =  json2dataframe("/Users/xinyilihuang/ParlAI/data/blended_skill_talk/valid.json")
test = json2dataframe("/Users/xinyilihuang/ParlAI/data/blended_skill_talk/test.json")


print(train.head)

## label encoding 
# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
print(train['label'].unique()) # alphabetic order
# Encode labels in column 'label'. 
train['label']= label_encoder.fit_transform(train['label']) 
print(train['label'].unique()) # alphabetic order
valid['label']= label_encoder.fit_transform(valid['label']) 
test['label']= label_encoder.fit_transform(test['label']) 


nclass = len(train['label'].unique())

print(train.groupby(['label']).size()) 
print(valid.groupby(['label']).size())
print(test.groupby(['label']).size()) 


'''
print(train.groupby(['label']).size()) 
manual_ind = train[train['label']==2].index
drop_ind = sample(list(manual_ind),19500) 
train = train.drop(drop_ind)
print(train.groupby(['label']).size()) 
'''


# tokenization
tok = spacy.load('en')
def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", str(text))
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]


#count the number of occurrences of each token in our corpus 
word2counts = Counter()
for sentence in train["input sentence"]:   
    word2counts.update(tokenize(sentence))
for sentence in valid["input sentence"]:   
    word2counts.update(tokenize(sentence))
for sentence in test["input sentence"]: 
    word2counts.update(tokenize(sentence))

#deleting infrequent words
print("number of words before removing infrequent ones:",len(word2counts.keys()))
for word in list(word2counts):
    if word2counts[word] < 2:
        del word2counts[word]
print("number of words after removing infrequent ones:",len(word2counts.keys()))

#mapping
word2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in word2counts:
    word2index[word] = len(words)
    words.append(word)


#create a vocabulary to index mapping and encode our text using this mapping
def encode_sentence(text, word2index, N=50):  # assuming that our inputs wont be more than 50 words long
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([word2index.get(word, word2index["UNK"]) for word in tokenized])  # get(key,value); value: A value to return if the specified key does not exist.
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

# apply encoding
train["encoded_sentence"] = train["input sentence"].apply(lambda x: np.array(encode_sentence(x,word2index)))
valid["encoded_sentence"] = valid["input sentence"].apply(lambda x: np.array(encode_sentence(x,word2index)))
test["encoded_sentence"] = test["input sentence"].apply(lambda x: np.array(encode_sentence(x,word2index)))

# split dataset
X_train = list(train["encoded_sentence"])  # an array and len of array(max_len)
y_train = list(train["label"])  
X_valid = list(valid["encoded_sentence"])  # an array and len of array(max_len)
y_valid = list(valid["label"]) 
X_test = list(test["encoded_sentence"])  # an array and len of array(max_len)
y_test = list(test["label"]) 

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## pytorch dataset
class toPytorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

trainDF = toPytorchDataset(X_train, y_train)  #  tensor(encoded_sentence), label, length 
validDF = toPytorchDataset(X_valid, y_valid)
testDF = toPytorchDataset(X_test, y_test)
print(testDF.__getitem__(0))

train_dl = DataLoader(trainDF, batch_size=batch_size, shuffle=True)  # put samples&features all together
val_dl = DataLoader(validDF, batch_size=batch_size)
test_dl = DataLoader(testDF, batch_size=batch_size)



# MODEL 
class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim) :
        super(LSTM_fixed_len,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        # readout layer: maps from hidden state space to tag_space
        self.linear = nn.Linear(hidden_dim, output_dim) 
        # to avoid overfitting
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, l):
        embeds = self.embeddings(sentence)
        embeds = self.dropout(embeds)
        embeds_pack = pack_padded_sequence(embeds, l, batch_first=True, enforce_sorted=False)  # for sentences with variable len
        lstm_out, (ht, ct) = self.lstm(embeds_pack)
        #consume the last_hidden_state of LSTM only
        y_pred = self.linear(ht[-1])  
        y_scores = self.softmax(y_pred)  
        return y_scores


## training process  
def train(model, epochs,lr):

    loss_ft = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    iter = 0
    for i in range(epochs):
        
        for x, y, l in train_dl: 
            x = x.long()
            #print(x.size())
            y = y.long()
            #print(y.size())
            y_pred = model(x,l)
            #print(y.max(),y.min())
            optimizer.zero_grad()  # Clear gradients w.r.t. parameters
            loss = loss_ft(y_pred, y)
            loss.backward() # compute loss 
            optimizer.step()  # update parameters 

            iter += 1

            if iter % 10 == 0:
                # Calculate Accuracy         
                val_accuracy,val_loss = evaluate(model, val_dl)
                print('Iteration: {}. Validation Loss:{}. Validation Accuracy: {}.'.format(iter, val_loss, val_accuracy))

    #return loss.item(), val_loss, val_accuracy

def evaluate(model, dataset):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0
    for x, y, l in dataset:
        #print(x,y,l)
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        #print(y_hat)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]  
        #print(pred)
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]

    eval_loss = sum_loss/total
    eval_acc = correct/total
    return eval_acc,eval_loss

vocab_size = len(words)
embedding_dim = 32
hidden_dim = 100
layer_dim = 5
EPOCHS = 50

#torch.manual_seed(124)
model_lstm =  LSTM_fixed_len(vocab_size, embedding_dim,hidden_dim, layer_dim, nclass)
train(model_lstm,EPOCHS,Learning_Rate)  
print(evaluate(model_lstm,test_dl))




