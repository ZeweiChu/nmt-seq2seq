import sys, os
import utils
import config
import code
import numpy as np
from models import *
import torch
from torch import optim
import pickle
import math
import nltk

def translate(model, data, en_dict, inv_en_dict, cn_dict, inv_cn_dict):
	for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(data):
		B, T = mb_x.shape

		mb_y = np.zeros((B, 1)).astype("int32")
		mb_y[:, 0] = cn_dict["BOS"]
		x = torch.from_numpy(mb_x).long()
		x_mask = torch.from_numpy(mb_x_mask).long()
		hidden = model.init_hidden(B)
		mb_y = torch.from_numpy(mb_y).long()
		if args.use_cuda:
			x = x.cuda()
			x_mask = x_mask.cuda()
			mb_y = mb_y.cuda()
		pred_y = model.translate(x, x_mask, mb_y, hidden, max_length=args.translation_max_length)

		if args.use_cuda:
			pred_y = pred_y.cpu()

		pred_y = pred_y.data.numpy()
		for i in range(B):
			en = ""
			for j in range(1, T):
				word = inv_en_dict[mb_x[i][j]]
				if word == "EOS":
					break
				en += inv_en_dict[mb_x[i][j]] + " "
			print(en)
			cn = ""
			for j in range(1, args.translation_max_length):
				word = inv_cn_dict[pred_y[i][j]]
				if word == "EOS":
					break
				cn += word
			print(cn)
		# code.interact(local=locals())
	

def eval(model, data, args, crit):
	total_dev_batches = len(data)
	correct_count = 0.
	total_loss = 0.
	total_num_words = 0.

	print("total %d" % total_dev_batches)
	total_num_words = 0.
	for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(data):
		
		batch_size = mb_x.shape[0]
		mb_x = torch.from_numpy(mb_x).long()
		mb_x_mask = torch.from_numpy(mb_x_mask).long()

		hidden = model.init_hidden(batch_size)
		mb_input = torch.from_numpy(mb_y[:,:-1]).long()
		mb_out = torch.from_numpy(mb_y[:, 1:]).long()
		mb_out_mask = torch.from_numpy(mb_y_mask[:, 1:])
		
		if args.use_cuda:
			mb_x = mb_x.cuda()
			mb_x_mask = mb_x_mask.cuda()
			mb_input = mb_input.cuda()
			mb_out = mb_out.cuda()
			mb_out_mask = mb_out_mask.cuda()
		
		with torch.no_grad():
			mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)

		num_words = torch.sum(mb_out_mask).item()
		batch_loss = crit(mb_pred, mb_out, mb_out_mask).item() 

		total_loss += batch_loss * num_words

		total_num_words += num_words
		

		mb_pred = torch.max(mb_pred.view(mb_pred.size(0) * mb_pred.size(1), mb_pred.size(2)), 1)[1]
		correct = (mb_pred.view(-1) == mb_out.view(-1)).float()

		correct_count += torch.sum(correct * mb_out_mask.contiguous().view(-1)).item()
	return correct_count, total_loss, total_num_words

def main(args):

	if os.path.isfile(args.vocab_file):
		en_dict, cn_dict, en_total_words, cn_total_words = pickle.load(open(args.vocab_file, "rb"))
	else:
		print("vocab file does not exit!")
		exit(-1)

	args.en_total_words = en_total_words
	args.cn_total_words = cn_total_words
	inv_en_dict = {v: k for k, v in en_dict.items()}
	inv_cn_dict = {v: k for k, v in cn_dict.items()}

	

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	else:
		print("model file does not exit!")
		exit(-1)

	if args.use_cuda:
		model = model.cuda()

	crit = utils.LanguageModelCriterion()

	test_en, test_cn = utils.load_data(args.test_file)
	args.num_test = len(test_en)
	test_en, test_cn = utils.encode(test_en, test_cn, en_dict, cn_dict)
	test_data = utils.gen_examples(test_en, test_cn, args.batch_size)
	
	translate(model, test_data, en_dict, inv_en_dict, cn_dict, inv_cn_dict)

	correct_count, loss, num_words = eval(model, test_data, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("test loss %s" % (loss) )
	print("test accuracy %f" % (acc))
	print("test total number of words %f" % (num_words))

if __name__ == "__main__":
	args = config.get_args()
	main(args)
