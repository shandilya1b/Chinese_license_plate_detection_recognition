#!/bin/bash

python3 run_inference.py \
	--object-weights "/ws/cortex/anpr/algorithms/Chinese_license_plate_detection_recognition/weights/plate_detect.pt"\
	--char-weights "/ws/cortex/anpr/algorithms/Chinese_license_plate_detection_recognition/weights/plate_rec_color.pth"\
	--out-dir out\
	--dataset-dir /ws/cortex/anpr/data/dataset-5/test/clean_images\
	--object-imgsz 1280\
	--char-imgsz 128\
	--device cpu
