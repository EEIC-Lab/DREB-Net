CUDA_VISIBLE_DEVICES=0 python demo.py \
--gpus 0 \
--input_res 1024 \
--dataset visdrone \
--arch DREB_Net \
--demo ./exp/test_images \
--inp_sharp_or_blur SB_deblur \
--load_model ./exp/detect/train/train_DREB_Net_model/model_last.pth \
--mode val \
--demo_save_path ./exp/demo_save