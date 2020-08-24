# [Sequence to sequence model for neural machine translation](https://github.com/ZeweiChu/nmt-seq2seq)

### Announcement
这个project是基于PyTorch 0.4.0版本写的。由于现在的PyTorch已经升级到了[1.6.0](https://pytorch.org/)版本，很多代码的写法不太符合最新的版本，所以欢迎大家重写我的代码然后send pull request！

### requirements: 
- python 3.6
- pytorch 0.4.0
- nltk
- tqdm

### What's included
- A vanilla seq2seq model implemented in PyTorch
- A keras version is stored in ```keras/```. The code is copied from [The Keras official tutorial](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)

### How to use
- Go to directory ```pytorch```
- To train the model, simply run
	./run.sh
- To test the model
	./run_test.sh
- To see what config options you have
	python main.py --help


### background 
- this repo tries to implement the paper [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), some details differ from the original paper, though. 
- we are using data from http://www.manythings.org/anki/

### 更新
由于PyTorch版本更新了好多次，最近重新写了[PyTorch的translation模型](pytorch/seq2seq.ipynb)。

### TODO
- add attention to the currently very basic model
- add beam search at inference 


### Bug report
If you find any bugs, please feel free to send an email to zeweichu@gmail.com , I will try to be responsive!
