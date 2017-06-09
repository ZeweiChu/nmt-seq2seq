model_dir="seq2seq"
if [ ! -d $model_dir ]; then
	mkdir $model_dir
fi
python main.py --train_file data/train_mini.txt --dev_file data/dev_mini.txt --test_file data/test_mini.txt
