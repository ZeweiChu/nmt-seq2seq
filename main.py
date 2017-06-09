import sys, os
import utils
import config
import code
import numpy as np
import pickle
import math
import nltk

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

	code.interact(local=locals())
