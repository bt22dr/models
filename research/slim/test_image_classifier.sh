DATA_DIR=/home/jysong/Documents/data/cdiscount/cdiscount_test_images
CKPT_DIR=/home/jysong/Documents/log/cdiscount/model
date
python test_image_classifier.py --ckpt_path=${CKPT_DIR} --dataset_dir=${DATA_DIR} --model_name=inception_resnet_v2 --image_size=180 > result.csv
date
