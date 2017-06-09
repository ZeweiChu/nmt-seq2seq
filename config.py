import argparse

def get_args():
    # Basics
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_cuda',
                        type=int,
                        default=0,
                        help='use cuda GPU or not 0|1')

    parser.add_argument('--model_file',
                        type=str,
                        default="model.th",
                        help='model file')

    parser.add_argument('--model',
                        type=str,
                        default="HingeModelCriterion",
                        help='choose the loss criterion')

    # Data file
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default=None,
                        help='Development file')

    parser.add_argument('--test_file',
                        type=str,
                        default=None,
                        help='test file')

    parser.add_argument('--vocab_file',
                        type=str,
                        default="vocab.pkl",
                        help='dictionary file')

    parser.add_argument('--vocab_size', 
                        type=int,
                        default=50000,
                        help='maximum number of vocabulary')

    parser.add_argument('--translation_max_length', 
                        type=int,
                        default=20,
                        help='translation max length')

    # Model details
    parser.add_argument('--embedding_size',
                        type=int,
                        default=300,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')

    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of RNN layers')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')

    parser.add_argument('--num_epoches',
                        type=int,
                        default=100,
                        help='Number of epoches')

    parser.add_argument('--eval_epoch',
                        type=int,
                        default=1,
                        help='Evaluation on dev set after K epochs')

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='Optimizer: sgd or adam (default) or rmsprop')

    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for SGD')

    return parser.parse_args()
