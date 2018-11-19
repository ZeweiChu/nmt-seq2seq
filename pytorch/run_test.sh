model_dir="seq2seq_saved_models"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi
python test.py --test_file ../data/en-cn/test.txt --batch_size 128 --num_epoches 20 --model_file $model_dir/model.th --learning_rate 0.01 --embedding_size 200 --hidden_size 200 --eval_epoch 1 --vocab_file $model_dir/vocab.pkl
