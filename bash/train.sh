ARCH=DREB_Net
EXP_ID=train_DREB_Net_model
DATASET=visdrone
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=../dataset/VisDrone/VisDrone-2019-DET-preprocess_ignore_black
BLUR_DATA_DIR=../dataset/VisDrone/VisDrone-2019-DET_blur/2_DeblurGAN/blur_image_ignore_black
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
CUDA_TRAIN_DEVICE=0,1,2,3

CUDA_VISIBLE_DEVICES=$CUDA_TRAIN_DEVICE python main.py \
--exp_id $EXP_ID \
--arch $ARCH \
--dataset $DATASET \
--inp_sharp_or_blur $INP_SHARP_OR_BLUR \
--sharp_data_dir $SHARP_DATA_DIR \
--blur_data_dir $BLUR_DATA_DIR \
--input_res 1024 \
--mode train \
--batch_size 16 \
--master_batch 4 \
--lr 1e-3  \
--num_epochs 200 \
--gpus $CUDA_TRAIN_DEVICE