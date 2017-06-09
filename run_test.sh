model_dir="seq2seq"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi
python test.py --train_file data/train_mini.txt --dev_file data/dev_mini.txt --test_file data/test_mini.txt --batch_size 128 --num_epoches 20 --model_file $model_dir/model.th --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --eval_epoch 1 --vocab_file $model_dir/vocab.pkl --use_cuda 1
