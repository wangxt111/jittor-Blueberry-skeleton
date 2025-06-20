skeleton_model_name=multihead
skeleton_model_name1=force_pct
skin_model_name=skin
python predict_skeleton.py --predict_data_list data/test_list_m.txt --data_root data --model_name $skeleton_model_name --pretrained_model '/home/ubuntu/jittor2025_skeleton/output/skeleton/multiheadsa+patience10/best_model.pkl' --predict_output_dir predict --batch_size 24
python predict_skeleton.py --predict_data_list data/test_list_v.txt --data_root data --model_name $skeleton_model_name1 --pretrained_model '/home/ubuntu/jittor2025_skeleton/output/sym/v/best_model.pkl' --predict_output_dir predict --batch_size 128