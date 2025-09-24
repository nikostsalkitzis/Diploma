# Diploma
To run the extract_feture.py do the following : python extract_features.py --dataset_path data/track2/ --out_features_path data/track2_features/
to run the train.py do this : 
>      python train.py \
>     --num_patients 8 \
>     --window_size 48 \
>     --save_path track2_lstmcnn \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --input_features 14 \
>     --cnn_channels 128 \
>     --lstm_hidden 32 \
>     --lstm_layers 4 \
>     --batch_size 16 \
>     --epochs 10 \
>     --device cuda
to run test.py do this : 
>      python test.py \
>     --num_patients 8 \
>     --window_size 48 \
>     --features_path data/track2_features/ \
>     --dataset_path data/track2/ \
>     --submission_path submissions_lstmcnn \
>     --load_path track2_lstmcnn/best_model.pth \
>     --scaler_path track2_lstmcnn/scaler.pkl \
>     --input_features 8 \
>     --cnn_channels 128 \
>     --lstm_hidden 32 \
>     --lstm_layers 4 \
>     --device cuda \
>     --mode val
For the newDiploma code for the train.py, run this:
```
     python train.py \
    --num_patients 8 \
    --window_size 48 \
    --save_path track2_rich_lstmcnn \
    --features_path data/track2_features/ \
    --dataset_path data/track2/ \
    --input_features 8 \
    --cnn_channels 128 \
    --cnn_blocks 3 \
    --lstm_hidden 32 \
    --lstm_layers 4 \
    --bidirectional \
    --attention \
    --batch_size 16 \
    --epochs 10 \
    --device cuda
```

and for the test.py:
```
python test.py \
    --num_patients 8 \
    --window_size 48 \
    --features_path data/track2_features/ \
    --dataset_path data/track2/ \
    --submission_path submissions_rich_lstmcnn \
    --load_path track2_rich_lstmcnn/best_model.pth \
    --scaler_path track2_rich_lstmcnn/scaler.pkl \
    --input_features 8 \
    --cnn_channels 128 \
    --cnn_blocks 3 \
    --lstm_hidden 32 \
    --lstm_layers 4 \
    --bidirectional \
    --attention \
    --device cuda \
    --mode val
```
To run the clustered elliptic envelope for the train.py do this:
```
python train.py \
--num_patients 8 \
--window_size 48 \
--save_path track2_lstmcnn \
--features_path data/track2_features/ \
--dataset_path data/track2/ \
--input_features 8 \
--cnn_channels 128 \
--lstm_hidden 32 \
--lstm_layers 4 \
--batch_size 16 \
--epochs 10 \
--device cuda \
--cluster_csv cluster.csv
```
and for the test.py :
```
python test.py \
--num_patients 8 \
--window_size 48 \
--features_path data/track2_features/ \
--dataset_path data/track2/ \
--submission_path submissions_lstmcnn \
--load_path track2_lstmcnn/best_model.pth \
--scaler_path track2_lstmcnn/scaler.pkl \
--input_features 8 \
--cnn_channels 128 \
--lstm_hidden 32 \
--lstm_layers 4 \
--device cuda \
--mode val \
--cluster_csv cluster.csv
```
