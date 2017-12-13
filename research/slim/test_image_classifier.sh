#DATA_DIR=/hadoop3/KCICC/data/cdiscount_test_images
#CKPT_DIR=/hadoop3/KCICC/log/tfmodel_e2e_1_tmp

#CUDA_VISIBLE_DEVICES='3' python test_image_classifier.py --checkpoint_path=${CKPT_DIR} --dataset_dir=${DATA_DIR} --batch_size=256 --num_preprocessing_threads=16

date
python test_image_classifier_for_nasnet2.py --ckpt_path=${CKPT_DIR} --dataset_dir=${DATA_DIR} --batch_size=256 > result2.csv
date
