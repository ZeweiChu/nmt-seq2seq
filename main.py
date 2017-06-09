import sys, os
import utils
import config
from models import *
import code
import numpy as np
import pickle
import math
import nltk
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm

def main(args):

	train_en, train_cn = utils.load_data(args.train_file)
	dev_en, dev_cn = utils.load_data(args.dev_file)
	args.num_train = len(train_en)
	args.num_dev = len(dev_en)

	# code.interact(local=locals())

	if os.path.isfile(args.vocab_file):
		en_dict, cn_dict, en_total_words, cn_total_words = pickle.load(open(args.vocab_file, "rb"))
	else:
		en_dict, en_total_words = utils.build_dict(train_en)
		cn_dict, cn_total_words = utils.build_dict(train_cn)
		pickle.dump([en_dict, cn_dict, en_total_words, cn_total_words], open(args.vocab_file, "wb"))

	args.en_total_words = en_total_words
	args.cn_total_words = cn_total_words
	inv_en_dict = {v: k for k, v in en_dict.items()}
	inv_cn_dict = {v: k for k, v in cn_dict.items()}

	train_en, train_cn = utils.encode(train_en, train_cn, en_dict, cn_dict)
	train_data = utils.gen_examples(train_en, train_cn, args.batch_size)

	dev_en, dev_cn = utils.encode(dev_en, dev_cn, en_dict, cn_dict)
	dev_data = utils.gen_examples(dev_en, dev_cn, args.batch_size)

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "EncoderDecoderModel":
		model = EncoderDecoderModel(args)

	if args.use_cuda:
		model = model.cuda()

	crit = utils.LanguageModelCriterion()

	learning_rate = args.learning_rate
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)
	
	total_num_sentences = 0.
	total_time = 0.
	for epoch in range(args.num_epochs):
		np.random.shuffle(train_data)
		total_train_loss = 0.
		total_num_words = 0.
		for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in tqdm(enumerate(train_data)):

			batch_size = mb_x.shape[0]
			total_num_sentences += batch_size
			mb_x = Variable(torch.from_numpy(mb_x)).long()
			mb_x_mask = Variable(torch.from_numpy(mb_x_mask)).long()
			hidden = model.init_hidden(batch_size)
			mb_input = Variable(torch.from_numpy(mb_y[:,:-1])).long()
			mb_out = Variable(torch.from_numpy(mb_y[:, 1:])).long()
			mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:]))

			if args.use_cuda:
				mb_x = mb_x.cuda()
				mb_x_mask = mb_x_mask.cuda()
				mb_input = mb_input.cuda()
				mb_out = mb_out.cuda()
				mb_out_mask = mb_out_mask.cuda()
			
			mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)

			loss = crit(mb_pred, mb_out, mb_out_mask)
			num_words = torch.sum(mb_out_mask).data[0]
			total_train_loss += loss.data[0] * num_words
			total_num_words += num_words
	
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		epoch += 1
		if epoch % args.num_epochs == 0:
			break 
		print("training loss: %f" % (total_train_loss / total_num_words))


if __name__ == "__main__":
	args = config.get_args()
	main(args)
