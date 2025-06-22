skeleton_model_name=pct
skeleton_model_name1=force_pct
skin_model_name=skin
python predict_skeleton.py --predict_data_list data/val_list_m.txt --data_root data --model_name $skeleton_model_name --pretrained_model '/home/ubuntu/jittor2025_skeleton/output/sym/m/best_model.pkl' --predict_output_dir predict/$skeleton_model_name --batch_size 128
python predict_skeleton.py --predict_data_list data/val_list_v.txt --data_root data --model_name $skeleton_model_name1 --pretrained_model '/home/ubuntu/jittor2025_skeleton/output/sym/v/best_model.pkl' --predict_output_dir predict/$skeleton_model_name --batch_size 128
python predict_skin.py --predict_data_list data/val_list.txt --data_root data --model_name $skin_model_name --pretrained_model '/home/ubuntu/jittor2025_skeleton/best_model1.pkl' --predict_output_dir predict/$skeleton_model_name --batch_size 8
python score.py --skeleton_model_name $skeleton_model_name --skin_model_name $skin_model_name