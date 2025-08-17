echo "[INFO] Lets start the compilation of our yolov3 object detection model."

vai_c_xir \
    --xmodel ./quantized_result/YOLOv3_int.xmodel \
    --arch ./shell_script/target_name.json \
    --output_dir ./compiled_result \
    --net_name "yolov3"

echo "[INFO] The compilation of yolov3 is completed!!!!!!!"