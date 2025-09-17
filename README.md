# Diploma
To run the extract_feture.py do the following : python extract_features.py --dataset_path data/track2/ --out_features_path data/track2_features/
to run the train.py do this : 
>      python train.py \
>     --num_patients 3 \
>     --window_size 48 \
>     --save_path track2_lstmcnn_128_32_2 \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --input_features 8 \
>     --cnn_channels 128 \
>     --lstm_hidden 32 \
>     --lstm_layers 2 \
>     --batch_size 16 \
>     --epochs 10 \
>     --device cuda
to run test.py do this : 
>      python test.py \
>     --num_patients 3 \
>     --window_size 48 \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --submission_path submissions/track2_lstmcnn \
>     --load_path track2_lstmcnn/best_model.pth \
>     --scaler_path track2_lstmcnn/scaler.pkl \
>     --input_features 8 \
>     --cnn_channels 128 \
>     --lstm_hidden 32 \
>     --lstm_layers 1 \
>     --device cuda \
>     --mode test
