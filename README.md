# Diploma
To run the extract_feture.py do the following : python extract_features.py --dataset_path data/track2/ --out_features_path data/track2_features/
to run the train.py do this : 
>      python train.py \
>     --num_patients 3 \
>     --window_size 48 \
>     --save_path track2_lstmcnn_128_16_4 \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --input_features 8 \
>     --cnn_channels 128 \
>     --lstm_hidden 16 \
>     --lstm_layers 4 \
>     --batch_size 16 \
>     --epochs 10 \
>     --device cuda
to run test.py do this : 
>      python test.py \
>     --num_patients 3 \
>     --window_size 48 \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --submission_path submissions_lstmcnn_128_16_4 \
>     --load_path track2_lstmcnn_128_16_4/best_model.pth \
>     --scaler_path track2_lstmcnn_128_16_4/scaler.pkl \
>     --input_features 8 \
>     --cnn_channels 128 \
>     --lstm_hidden 16 \
>     --lstm_layers 4 \
>     --device cuda \
>     --mode val
