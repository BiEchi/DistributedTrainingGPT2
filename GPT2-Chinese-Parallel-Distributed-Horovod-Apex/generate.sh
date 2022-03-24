python3 generate.py \
  --device 0 \
  --length 15000 \
  --tokenizer_path cache/vocab_user.txt \
  --model_path model/final_model \
  --prefix "她" \
  --nsamples 5 \
  --topp 1 \
  --temperature 1.0
