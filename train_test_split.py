
from sklearn.model_selection import train_test_split

data = []
num_examples = 0
with open("data/cmn.txt", 'r') as f:
    for line in f:
        data.append(line)

train, dev = train_test_split(data, test_size=0.2)
dev, test = train_test_split(dev, test_size=0.5)

with open("data/train.txt", 'w') as f:
    for line in train:
    	f.write(line)

with open("data/dev.txt", 'w') as f:
    for line in dev:
    	f.write(line)

with open("data/test.txt", 'w') as f:
    for line in test:
    	f.write(line)
