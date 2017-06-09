model_dir="seq2seq"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi
python main.py --train_file data/train.txt --dev_file data/dev.txt --test_file data/test.txt --batch_size 64 --num_epoches 20 --model_file $model_dir/model.th --model EncoderDecoderModel --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --eval_epoch 1 --vocab_file $model_dir/vocab.pkl --use_cuda 1
