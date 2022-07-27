#!/usr/bin/env python
# coding: utf-8

# # Transformer from scratch --------------------

# Transformer has the encoder and decoder type structure, it main focus is on the attention. It compute the self attention. 
# for sentiment analysis we will use only the encoder part of the transformer. The tranformer takes the whole sentence as input, so they are very fast but it forget the sequence of the sentences for this we use positonal embedding. Then it computes the self attention and pass the input to linear layers.

# In[1]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[2]:


# Hyperparameters ------------------

seq_len = 50
embed_dim = 100
n_heads = 5
expansion = 4
batch_size = 64
num_layers = 2


# In[3]:


# Class to calculate the multi-head self attention ----------------------

class Attention(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.dim = self.embed_dim // self.n_heads
        self.key = nn.Linear(self.dim, self.dim)
        self.query = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        
        
    def forward(self, key, query, value):
        batch, seq, embed = key.shape
        
        multi_key = key.reshape(batch, seq, self.n_heads, self.dim)
        multi_query = query.reshape(batch, seq, self.n_heads, self.dim)
        multi_value = value.reshape(batch, seq, self.n_heads, self.dim)
        
        key_out = self.key(multi_key)
        query_out = self.query(multi_query)
        value_out = self.value(multi_value)

        energy = torch.einsum('bqhd, bkhd->bqhk', query_out, key_out)
        softmax = torch.softmax(energy/self.embed_dim**(1/2), dim=3)
        attention = torch.einsum('bqhk, bvhe -> bqhe', softmax, value_out)
        
        return attention.reshape(batch, seq, -1)
        


# In[4]:


# Check of multi-head self attention ---------------

key = torch.randn((16, 10, 50))
query = torch.randn((16, 10, 50))
value = torch.randn((16, 10, 50))


# In[5]:


a = Attention(5, 50)
out = a(key, query, value)
out.shape


# In[6]:


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, expansion):
        super().__init__()
        self.attention = Attention(n_heads, embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(),
            nn.Linear(embed_dim* expansion, embed_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        att = self.attention(x, x, x)
        out = self.linear(att)
        return out


# In[7]:


e = Encoder(50, 5, 4)
out = e(key)
out.shape


# In[45]:


# Model to Predict the sentiment --------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len * embed_dim, 1200),
            nn.ReLU(),
            nn.Linear(1200, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        return self.model(x)
    


# In[46]:


# Transformer -----------------------------------

class Transformer(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers, expansion, vocab_size, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList()
        self.final = Model()
        
        for i in range(num_layers):
            self.layers.append(Encoder(embed_dim, n_heads, expansion))
        
    def forward(self, x):
        batch,seq_len = x.shape
        
        positions = torch.arange(0, seq_len).expand(batch, seq_len)
        embed = self.embed(x)
        postion_embed = self.position(positions)
        
        x = embed + postion_embed
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(batch, -1)

        return self.final(x)
    


# Lets Change the given data into the format that can be processed by the transformer, preprocessing of the Data 

# In[10]:


# Read the Dataset ----------------------------

df = pd.read_csv("D:/Datasets/NLP/sentiment/airline_sentiment_analysis.csv")
df.head()


# In[11]:


# preprocess the Dataset -------------------

df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[12]:


le = LabelEncoder()
df['airline_sentiment'] = le.fit_transform(df['airline_sentiment'])
df.head()


# In[13]:


# Lets select the 500 examples of both positive and negative

p_df = df[df['airline_sentiment']==1].sample(500)
n_df = df[df['airline_sentiment']==0].sample(500)

df = pd.concat([p_df, n_df])


# In[14]:


df.shape


# In[15]:


from torch.utils.data import Dataset, DataLoader


# In[16]:


# Create a Dataset ----------------

class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, item):
        return self.x[item], self.y[item]
    
    def __len__(self):
        return self.x.shape[0]


# In[17]:


# Tokenizer ----------

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

tokenizer = RegexpTokenizer("[\w']+")


# In[29]:


# Create a new-df for better training -------------------

new_df = df.copy()
stop = set(stopwords.words('english'))

for i in range(df.shape[0]):
    sent = df.iloc[i, 1]
    sent = re.sub('@[\w]+', '', sent)
    
    words = ''
    for word in tokenizer.tokenize(sent):
        if word not in stop:
            if word != ' ':
                words += (word.lower() +' ')
            
    new_df.iloc[i, 1] = words


# In[30]:


new_df.head()


# In[20]:


# Split data into x and y

x_ = new_df['text']
y_ = new_df['airline_sentiment']


# In[21]:


# Split data into x_train and y_train -------------------

x_train, x_test, y_train, y_test = train_test_split(x_, y_, train_size=0.8)

x_train.shape, x_test.shape


# In[27]:


# Create the dictionry---------------

def create_dictionary(train_data, min_freq, d_size=100000000000):
    d = {'PAD':1, 'UNK':2}
    f = {}
    i = 3
    
    for data in train_data:
        for word in tokenizer.tokenize(data):
            if word not in f:
                f[word] = 0
                
            f[word] += 1
            
            if f[word] == min_freq and len(d) < d_size:
                if word not in d:
                    d[word] = i
                    i += 1
    
    return d


# In[32]:


d = create_dictionary(x_train, 1)


# In[51]:


len(d)


# In[33]:


# Tokenize the sentences and padd them ---------------------------

def tokenize_and_padd(data, vocab, sent_len=50):
    all_data = []
    for sent in data:
        temp = []
        for word in tokenizer.tokenize(sent):
            if word in vocab:
                temp.append(vocab[word])
                
            else:
                temp.append(2)
        
        extra = sent_len - len(temp)
        
        if extra < 0:
            temp = temp[:sent_len]
            
        extra = sent_len - len(temp)
        
        while extra > 0:
            temp.append(1)
            extra -= 1
        
        all_data.append(temp)
    return all_data


# In[34]:


# x_train and x-test in the form of int -------------

train_data = tokenize_and_padd(x_train, d) 
test_data = tokenize_and_padd(x_test, d)


# In[59]:


# Lets create the dataset off the x_train and x_test -------------

t_data = Data(np.array(train_data), np.array(y_train))
te_data = Data(np.array(test_data), np.array(y_test))


# In[60]:


# Lets divide the data into batches ------------------------

train_batch = DataLoader(t_data, batch_size)
test_batch = DataLoader(te_data, batch_size)


# In[68]:


# Some Model training Hyperparameters --------------

epochs = 15
trans = Transformer(embed_dim, n_heads, num_layers, expansion, len(d), seq_len)
lr = 0.004
los = torch.nn.CrossEntropyLoss()
opti = torch.optim.Adam(trans.parameters(), lr)


# In[69]:


es = []
ls = []
ac = []


# In[70]:


# Function to calculate the accuracy -----------------------------

def accuracy(y_hat, y):
    correct = 0
    tot = 0
    for i in range(len(y)):
        if y[i] == torch.argmax(y_hat[i]):
            correct += 1
        tot += 1
    
    return correct/tot


# In[71]:


# Training of the Transformer ------------------

for epoch in range(epochs):
    loss = []
    acc = []
    for i, (txt,  label) in enumerate(train_batch):
        if i == len(train_batch)-1:
            continue
#         print("yes")
        y_hat = trans(txt)
        
        acc.append(accuracy(y_hat, label))
        l = los(y_hat, label.long())
        
        loss.append(l.item())
        l.backward()
    
        opti.step()
        opti.zero_grad()
    
    es.append(epoch)
    a = np.array(acc)
    b = np.array(loss)
    
    print(epoch, np.mean(b))
    ac.append(np.mean(a))
    ls.append(np.mean(b))


# In[72]:


print(es, ls, ac)


# In[74]:


# PLot the Results --------------

sns.lineplot(es, ls)
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[76]:



sns.lineplot(es, ac)
plt.title("Epoch vs Acuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# In[138]:


sent = "i love you"


# In[144]:


input = []
for word in tokenizer.tokenize(sent):
    word = word.lower()
    if word not in d:
        input.append(2)
        
    else:
        input.append(d[word])


# In[145]:


input


# In[146]:


n = len(input)
temp = seq_len - n

while temp:
    input.append(1)
    temp -= 1


# In[147]:


input = torch.Tensor(input)
input = input.unsqueeze(0)
input.shape


# In[148]:


res = None
with torch.no_grad():
    out = trans(input.long())
    emotion = torch.argmax(out)
    print(out, emotion)
    if emotion == 1:
        res = "Positive"
    else:
        res = 'Negative'


# In[137]:


res


# In[ ]:




