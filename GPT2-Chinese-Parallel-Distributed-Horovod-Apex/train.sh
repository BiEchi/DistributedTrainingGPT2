horovodrun -np 4 -H localhost:4 \
python train.py \
  --fp16 \
  --fp16_opt_level='O1' \
  --max_grad_norm=1.0 \
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
