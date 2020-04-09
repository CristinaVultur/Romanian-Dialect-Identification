from tensorflow import keras
import numpy as np
import csv

ids = []
train_samples = []
#pun datele de train si validare impreuna pentru a crea tot vocabularul
with open("train_samples.txt", encoding="utf-8", mode="r") as f:
    for line in f.readlines():
        id, text = line.split("\t")
        ids.append(id)
        train_samples.append(text)

with open("validation_samples.txt", encoding="utf-8", mode="r") as f:
    for line in f.readlines():
        id, text = line.split("\t")
        ids.append(id)
        train_samples.append(text)

train_without_sw = []
max_len = 0
vocabulary_set = set()  #set pentru a nu se repeta cuvintele

#tranform textele in tokens si creez vocabularul

for text in train_samples:
    word_token = text.split()
    for w in word_token:
        vocabulary_set.add(w)
    train_without_sw.append(word_token)
    if max_len < len(word_token):
        max_len = len(word_token)

#max_len este lungimea celui mai lung text

#imi creez un vocabular "invers" ( fiecarui cuvand ii asociez un numar unic)
vocab_size = len(vocabulary_set)
# word index used for encoding
word_index = {k: (v + 3) for k, v in zip(vocabulary_set, range(1, 1 + vocab_size))}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2   # unknown
word_index["<UNUSED>"] = 3

#transformam cuvintele din text prin inlocuirea lor cu numarul unic asociat fiecaruia in vocabularul "invers"
def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index:
            encoded.append(word_index[word])
    return encoded

train_samples = []
for line in train_without_sw:
    train_samples.append(review_encode(line))

vocab_size = len(vocabulary_set)

test_samples = []
test_ids =[]

#preprocesez si datele de test la fel ca cele de train

with open("test_samples.txt", encoding="utf-8", mode="r") as f:
    for line in f.readlines():
        id, text = line.split("\t")
        test_ids.append(id)
        word_token = text.split()
        if max_len < len(word_token):
            max_len = len(word_token)
        encode = review_encode(word_token)
        test_samples.append(encode)

#deoarece avem nevoie de date cu aceeasi lungime adaug la fiecare text padding pana la lungimea max_len (lungimea celui mai lung sir si in test si in train)

train_samples = keras.preprocessing.sequence.pad_sequences(train_samples, value=word_index["<PAD>"] , padding="post", maxlen = max_len)
train_labels = []

# etichetele pentru datele de train

with open("train_labels.txt", mode="r") as f:
    for line in f.readlines():
        id, label = line.split("\t")
        train_labels.append(int(label))

with open("validation_labels.txt", mode="r") as f:
    for line in f.readlines():
        id, label = line.split("\t")
        train_labels.append(int(label))

#imi creez reteaua neuronala

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size+5,12)) #layer embedding pe 16 dimensiuni care asociaza fiecarui cuvant
# cate un vector si pe masura ce invata muta acei vectori
model.add(keras.layers.GlobalAveragePooling1D()) #compactam cele 12 dimensiuni intr-una singura
model.add(keras.layers.Dense(12, activation='relu')) #layer cu 12 neuroni (nr egal cu cel al dimensiunilor)  ce are functia de activare de tip relu
model.add(keras.layers.Dense(1,activation='sigmoid'))#layer cu un neuron ce are functia de activare sigmoid pentru ca vreau ca outputul meu sa fie un nr intre 0 si 1

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#antrenam modelul, 10% din totatul datelor sunt pt validare,
# vreau ca dupa o epoca datele sa se schimbe cu altele (shuffle =true),
# impart datele in 9 si trec prin 18 epoci
fitModel = model.fit(train_samples, np.array(train_labels), epochs=16, shuffle='True', batch_size=8, validation_split=0.1)# 9 si 18 cel mai bine

test = keras.preprocessing.sequence.pad_sequences(test_samples, value=word_index["<PAD>"], padding="post", maxlen=max_len)
predict = model.predict(test)

with open('predictions.csv', mode='w',newline='') as predictions:
    writer = csv.writer(predictions, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'label'])
    for i, pred in zip(test_ids,predict):
        writer.writerow([i, int(round(pred[0]))])