# Dataset 
2776584 total
 - 2706584 train set
 -   70000 valid set

# Step 1 Fine-tuning
## Training
```
WORK_DIR=~/Documents/visual_search_slim
RSC_DIR=~/Documents
LOG_DIR=${RSC_DIR}/log
TRAIN_DIR=${LOG_DIR}/cdiscount/train
EVAL_DIR=${LOG_DIR}/cdiscount/eval
DATA_DIR=${RSC_DIR}/data/cdiscount
DATA_NAME=cdiscount_0_2K
CKPT_DIR=${RSC_DIR}/checkpoints/nasnet-a_large_04_10_2017
CKPT_PATH=${CKPT_DIR}/model.ckpt

cd $WORK_DIR
cd tensorflow_models/research/slim
python download_and_convert_data.py --dataset_name ${DATA_NAME} --dataset_dir ${DATA_DIR}

python train_image_classifier.py \
--train_dir=${TRAIN_DIR} \
--dataset_dir=${DATA_DIR} \
--dataset_name=${DATA_NAME} \
--dataset_split_name=train \
--model_name=nasnet_large \
--checkpoint_path=${CKPT_PATH} \
--preprocessing_name=inception \
--clone_on_cpu=True \
--moving_average_decay=0.999 \
--checkpoint_exclude_scopes=aux_11/aux_logits/FC,final_layer/FC \
--trainable_scopes=aux_11/aux_logits/FC,final_layer/FC
```

## Evaluation
```
CUDA_VISIBLE_DEVICES='' python eval_image_classifier_loop.py \
 --alsologtostderr \
 --checkpoint_path=${TRAIN_DIR} \
 --dataset_dir=${DATA_DIR} \
 --eval_dir=${EVAL_DIR} \
 --dataset_name=${DATA_NAME} \
 --dataset_split_name=validation \
 --max_num_batches=180 \
 --batch_size=8 \
 --eval_interval_secs=3600 \
 --model_name=nasnet_large
```

## Tensorboard
```
tensorboard --logdir=$LOG_DIR --port=6008
```

# Step 2 End-to-End Learning
