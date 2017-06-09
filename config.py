import argparse

def get_args():
    parser = argparse.ArgumentParser()


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

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')
  
    return parser.parse_args()
