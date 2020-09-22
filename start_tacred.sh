export MODELNAME=loss2mask_layerdecay
export MODELPATH=trained_model/text_loss2_mask_layer_decay
export STEP=70000
export CUDA=7
CUDA_VISIBLE_DEVICES=$CUDA python code/run_tacred_trans.py \
  --do_train \
  --do_eval \
  --eval_test \
  --data_dir data/tacred \
  --data_cache_dir cache/tacred \
  --model $MODELPATH/checkpoint-$STEP \
  --vocab $MODELPATH \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir output_$MODELNAME_$STEP
