#!/bin/bash

random_flag=True   # define whether to use random stir (random pick random place) or random pick model place
model_path=None   # if random pick model place, define the model path, else set 'None'
rgb_flag=True    # define whether to collect rgb image
gif_flag=false   # define whether to collect gif image

collect_epoch=2  # define the number of epochs to collect data


TARGET_DIR="/media/eric/T31/Data/Basket/Retrieve"


if [ ! -d "$TARGET_DIR" ]; then
    echo "No exists dir"
    mkdir -p "$TARGET_DIR/point_cloud"
    mkdir -p "$TARGET_DIR/rgb"
    mkdir -p "$TARGET_DIR/gif"
    touch "$TARGET_DIR/Record.txt"
    echo "Make $TARGET_DIR"
fi

for ((i=0; i<collect_epoch; i++))
do
    # ~/.local/share/ov/pkg/isaac-sim-2023.1.1/python.sh Env_Data_Collection/basket_retrieve.py $random_flag $model_path $rgb_flag $gif_flag
    ~/.local/share/ov/pkg/isaac-sim-2023.1.1/python.sh /home/eric/Documents/dyros/garment_Fusion/all_scene.py $random_flag $model_path $rgb_flag $gif_flag
    sleep 5
done
