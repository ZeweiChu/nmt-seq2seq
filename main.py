import sys, os
import utils
import config
import code
import numpy as np
from models import *
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import MSELoss
from tqdm import tqdm
import pickle
import math
import nltk

def eval(model, data, args, crit):
	total_dev_batches = len(data)
	correct_count = 0.
	# bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	loss = 0.
	total_num_words = 0.

	print("total %d" % total_dev_batches)
	total_num_words = 0.
	for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in enumerate(data):
		# code.interact(local=locals())
		batch_size = mb_x.shape[0]
		mb_x = Variable(torch.from_numpy(mb_x), volatile=True).long()
		mb_x_mask = Variable(torch.from_numpy(mb_x_mask), volatile=True).long()
		hidden = model.init_hidden(batch_size)
		mb_input = Variable(torch.from_numpy(mb_y[:,:-1]), volatile=True).long()
		mb_out = Variable(torch.from_numpy(mb_y[:, 1:]), volatile=True).long()
		mb_out_mask = Variable(torch.from_numpy(mb_y_mask[:, 1:]), volatile=True)
		if args.use_cuda:
			mb_x = mb_x.cuda()
			mb_x_mask = mb_x_mask.cuda()
			mb_input = mb_input.cuda()
			mb_out = mb_out.cuda()
			mb_out_mask = mb_out_mask.cuda()
		
		mb_pred, hidden = model(mb_x, mb_x_mask, mb_input, hidden)
		num_words = torch.sum(mb_out_mask).data[0]
		loss += crit(mb_pred, mb_out, mb_out_mask).data[0] * num_words

		total_num_words += num_words
		

		mb_pred = torch.max(mb_pred.view(mb_pred.size(0) * mb_pred.size(1), mb_pred.size(2)), 1)[1]
		# code.interact(local=locals())
		correct = (mb_pred == mb_out).float()

		correct_count += torch.sum(correct * mb_out_mask.contiguous().view(mb_out_mask.size(0) * mb_out_mask.size(1), 1)).data[0]
	return correct_count, loss, total_num_words

def main(args):

	# load sentences (English and Chinese words)
	train_en, train_cn = utils.load_data(args.train_file)
	dev_en, dev_cn = utils.load_data(args.dev_file)
	args.num_train = len(train_en)
	args.num_dev = len(dev_en)

	# build English and Chinese dictionary
	if os.path.isfile(args.vocab_file):
		en_dict, cn_dict, en_total_words, cn_total_words = pickle.load(open(args.vocab_file, "rb"))
	else:
		en_dict, en_total_words = utils.build_dict(train_en)
		cn_dict, cn_total_words = utils.build_dict(train_cn)
		pickle.dump([en_dict, cn_dict, en_total_words, cn_total_words], open(args.vocab_file, "wb"))

	args.en_total_words = en_total_words
	args.cn_total_words = cn_total_words
	# index to words dict
	inv_en_dict = {v: k for k, v in en_dict.items()}
	inv_cn_dict = {v: k for k, v in cn_dict.items()}

	# encode train and dev sentences into indieces
	train_en, train_cn = utils.encode(train_en, train_cn, en_dict, cn_dict)
	# convert to numpy tensors
	train_data = utils.gen_examples(train_en, train_cn, args.batch_size)

	dev_en, dev_cn = utils.encode(dev_en, dev_cn, en_dict, cn_dict)
	dev_data = utils.gen_examples(dev_en, dev_cn, args.batch_size)

	# code.interact(local=locals())

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "EncoderDecoderModel":
		model = EncoderDecoderModel(args)

	if args.use_cuda:
		model = model.cuda()

	crit = utils.LanguageModelCriterion()

	print("start evaluating on dev...")
	correct_count, loss, num_words = eval(model, dev_data, args, crit)

	loss = loss / num_words
	acc = correct_count / num_words
	print("dev loss %s" % (loss) )
	print("dev accuracy %f" % (acc))
	print("dev total number of words %f" % (num_words))
	best_acc = acc

	learning_rate = args.learning_rate
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)
	
	total_num_sentences = 0.
	total_time = 0.
	for epoch in range(args.num_epoches):
		np.random.shuffle(train_data)
		total_train_loss = 0.
		total_num_words = 0.
		for idx, (mb_x, mb_x_mask, mb_y, mb_y_mask) in tqdm(enumerate(train_data)):

			batch_size = mb_x.shape[0]
			total_num_sentences += batch_size
			# convert numpy ndarray to PyTorch tensors and variables
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
		print("training loss: %f" % (total_train_loss / total_num_words))

		# evaluate every eval_epoch
		if (epoch+1) % args.eval_epoch == 0:
			

			print("start evaluating on dev...")
	
			correct_count, loss, num_words = eval(model, dev_data, args, crit)

			loss = loss / num_words
			acc = correct_count / num_words
			print("dev loss %s" % (loss) )
			print("dev accuracy %f" % (acc))
			print("dev total number of words %f" % (num_words))

			# save model if we have the best accuracy
			if acc >= best_acc:
				torch.save(model, args.model_file)
				best_acc = acc

				print("model saved...")
			else:
				learning_rate *= 0.5
				optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)

			print("best dev accuracy: %f" % best_acc)
			print("#" * 60)

	# load test data
	test_en, test_cn = utils.load_data(args.test_file)
	args.num_test = len(test_en)
	test_en, test_cn = utils.encode(test_en, test_cn, en_dict, cn_dict)
	test_data = utils.gen_examples(test_en, test_cn, args.batch_size)

	# evaluate on test
	correct_count, loss, num_words = eval(model, test_data, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("test loss %s" % (loss) )
	print("test accuracy %f" % (acc))
	print("test total number of words %f" % (num_words))

	# evaluate on train
	correct_count, loss, num_words = eval(model, train_data, args, crit)
	loss = loss / num_words
	acc = correct_count / num_words
	print("train loss %s" % (loss) )
	print("train accuracy %f" % (acc))


if __name__ == "__main__":
	args = config.get_args()
	main(args)
