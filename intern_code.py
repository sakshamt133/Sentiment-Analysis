#!/usr/bin/env python
# coding: utf-8

# Creating a sentiment analysis which helps to find the sentiment of a review 

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import gensim
import string
import nltk
import re
from nltk.corpus import stopwords


# In[2]:


batch_size = 64
epochs = 8
lr = 0.001
input_size = 50
hidden_size = 720
dictionary_size = 10000


# In[3]:


# Read the Dataset ----------------

df = pd.read_csv("D:/Datasets/NLP/sentiment/airline_sentiment_analysis.csv")
df.head()


# Preprocessing of the Data 
# 

# In[4]:


# Drop the unnamed column -----------------

df.drop(['Unnamed: 0'], axis=1, inplace=True)
# df.head()


# In[5]:


# Remove the Stop Words and punctuation from the sentence ----------------------

words = []
tokenizer = nltk.RegexpTokenizer("[\w']+")
stop = set(stopwords.words('english'))
new_df = df.copy()

for i in range(df.shape[0]):
    
#     To remove the @username from the sentence ---------------
    sent = df.iloc[i, 1]
    sent = re.sub('@[\w]+', '', sent)
    
#     Remove the stop words from the sentences ----------------
    words = ''
    for word in tokenizer.tokenize(sent):
        if word not in stop:
            word = re.sub('[\d]+', '', word)
            if word != '':
                words+= word.lower() + ' '
      
    new_df.iloc[i, 1] = words


# In[6]:


# How many positive and negative examples are there ---------------------------

new_df['airline_sentiment'].value_counts()


# Our dataset is highly imbalanced ------------------------------------------
# To make it balanced lets rendomly select some of the negative examples ----------------------

# In[7]:


p_df = new_df[new_df['airline_sentiment']=='positive']
n_df = new_df[new_df['airline_sentiment']=='negative'].sample(2400)

df1 = pd.concat([p_df, n_df])


# In[8]:


# Check the no.of values in the new dataset ---------------

df1['airline_sentiment'].value_counts()


# In[9]:


# Change the value of postive and negative to 1 and 0

df1['airline_sentiment'].replace('positive', 1, inplace=True)
df1['airline_sentiment'].replace('negative', 0, inplace=True)

df1.head()


# In[10]:


# Split the data into x_train and x_test ------------------

from sklearn.model_selection import train_test_split

x_train_, x_test_, y_train_, y_test_ = train_test_split(df1['text'], df1[['airline_sentiment']])


# In[11]:


# a function to create a dictionary and map all our words to dictionary ---------------------

def create_dictionary_and_map(x_train,  x_test, dictionary_len):
    sent_len = 100
    dictionary = {}
    s_to_i = {'<PAD>': 0, 'UNK':1}
    i = 2
    
    for sent in x_train:
        for word in tokenizer.tokenize(sent):
            if word not in dictionary and len(dictionary) < dictionary_len:
                dictionary[word] = i
                s_to_i[word] = i
                i += 1
 
    x_training = []
    x_testing = []
    for sent in x_train:
        temp = []
        for word in tokenizer.tokenize(sent):
            if word in dictionary:
                temp.append(dictionary[word])
            else:
                temp.append(1)
                
        n = len(temp)
        if n > sent_len:
            temp = temp[:100]
            
        while n < sent_len:
            temp.append(0)
            n += 1
        
        x_training.append(temp)
        
    for sent in x_test:
        temp = []
        for word in tokenizer.tokenize(sent):
            if word in dictionary:
                temp.append(dictionary[word])
            else:
                temp.append(1)
            
        n = len(temp)
        if n > sent_len:
            temp = temp[:100]
            
        while n < sent_len:
            temp.append(0)
            n += 1
        
        x_testing.append(temp)
    
    return np.array(x_training), np.array(x_testing), dictionary
                


# In[12]:


# To check the dictionary and maping function ------------------------

x_train, x_test, vocab = create_dictionary_and_map(x_train_, x_test_, dictionary_size)


# In[13]:


len(vocab)


# In[16]:


# Lets create the Dataset class for pytorch --------------------

class Data(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
    
    def __getitem__(self, index):
        return (self.data[index], self.label[index])
    
    def __len__(self):
        return self.data.shape[0]


# In[17]:


# Divide the training data into batches ----------------------

train_data = Data(x_train, np.array(y_train_))

train_batches = DataLoader(
    train_data, 
    batch_size=batch_size
)


# In[111]:


# for (t, label) in train_batches:
#     print(t)
#     print(t.shape, label.shape)
#     break


# In[18]:


# Lets divide testing data into batches -----------------------

test_data = Data(x_test, np.array(y_test_))

test_batches = DataLoader(
    test_data, 
    batch_size=batch_size,
    num_workers=4
)


# Lets Create the Model 

# In[15]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# In[20]:


# Create the Model using LSTM and Linear layer ------------

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        embed = self.embed(x)
        
        lstm, (hidden,_) = self.lstm(embed)
        
        out = self.dropout(hidden.squeeze())
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
 
        return out


# In[21]:


# Declare the Model -------------------

lstm = LSTM_Model(input_size, hidden_size, len(vocab), 1)
lstm


# Train the Model ------------------

# In[22]:


# To calculate accuracy of the Model ---------------

def accu(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label).item()


# In[23]:


es = []
lr = 0.008
acc = []
losses = []
opti = torch.optim.Adam(lstm.parameters(), lr)
l_fn = nn.CrossEntropyLoss()


# In[24]:


# Training of the Model ------------------------------

for epoch in range(epochs):
    l = []
    t_acc = 0
    for i, (text, label) in enumerate(train_batches):
        if i == len(train_batches) -1:
            break
            
        y_hat = lstm(text)
  
        loss = l_fn(y_hat, label.squeeze().long())
        l.append(loss.item())
        
        accuracy = accu(y_hat,label)
        t_acc += accuracy
        loss.backward()

        opti.step()
        opti.zero_grad()
        
    l = np.array(l)
    print(epoch, np.mean(l))
    es.append(epoch)
    losses.append(np.mean(l))
    acc.append(t_acc/len(train_batches.dataset))
    


# In[25]:


# PLoting Epochs vs Losses --------------------------

sns.lineplot(es, losses)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.show()


# In[26]:


# Plotting the Epochs vs Accruacy
print(len(es), len(acc))
sns.lineplot(es, acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# In[27]:


# Save the Model --------------------------

torch.save({
    'vocab': vocab,
    'model_state_dict':lstm.state_dict(),
    'opti_state_dict':opti.state_dict(),
    'tokenizer':tokenizer,
}, 'model.pth')


# In[28]:


sent = "Nice to meet you"

pred = []
for word in tokenizer.tokenize(sent):
    if word not in stop:
        if word in vocab:
            pred.append(vocab[word])
        else:
            pred.append(1)


# In[29]:


pred = torch.Tensor(pred)
pred = pred.unsqueeze(0)
with torch.no_grad():
    out = lstm(pred.long())
    
out


# In[31]:


torch.argmax(out).item()


# In[ ]:




