ARCH=DREB_Net
EXP_ID=test_DREB_Net_model
DATASET=visdrone
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=../dataset/VisDrone/VisDrone-2019-DET-preprocess_ignore_black
BLUR_DATA_DIR=../dataset/VisDrone/VisDrone-2019-DET_blur/2_DeblurGAN/blur_image_ignore_black
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
CUDA_VAL_DEVICE=1

CUDA_VISIBLE_DEVICES=$CUDA_VAL_DEVICE python test.py \
--exp_id $EXP_ID \
--arch $ARCH \
--dataset $DATASET \
--inp_sharp_or_blur $INP_SHARP_OR_BLUR \
--sharp_data_dir $SHARP_DATA_DIR \
--blur_data_dir $BLUR_DATA_DIR \
--input_res 1024 \
--gpus $CUDA_VAL_DEVICE \
--mode test \
--fix_res \
--load_model $BEST_MODEL \
--flip_test
