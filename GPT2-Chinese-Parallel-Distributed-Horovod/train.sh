horovodrun -np 8 -H hal15:4,hal16:4 \
python train.py \
  --model_config config/model_config.json \
  --tokenized_data_path data/tokenized/ \
  --tokenizer_path cache/vocab_user.txt \
  --raw_data_path data/train.json \
  --epochs 4000 \
  --batch_size 4 \
  --log_step 200 \
  --stride 512 \
  --output_dir model/ \
  --device 0,1,2,3 \
  --num_pieces 100 \
  --raw # turn this on if you want to pretrain
