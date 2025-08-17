echo "[INFO] Lets start the quantization of our YOLOv3 object detection model."

#Calibration
python3 code/yolo_quantize.py --mode 'calib'

#Testing
# python3 code/yolo_quantize.py --mode 'test'

#Deployment
python3 code/yolo_quantize.py --mode 'test' --device 'cpu' --deploy


echo "[INFO] The quantization of our YOLOv3 is completed!!!!!!!!!!!!"