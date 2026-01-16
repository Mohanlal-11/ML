# Float model check
```bash
python3 Deployment/yolo_quantization.py --mode float --calib_data /calib_data/

# Calibration
```bash
python3 Deployment/yolo_quantization.py --mode calib --calib_data /calib_data/
```

# Deployment
```bash
python3 Deployment/yolo_quantization.py --mode test --device cpu --deploy --calib_data /calib_data/
```

# Compilation
```bash
vai_c_xir --xmodel Deployment/quantized_result_nano/YOLOv5_int.xmodel --arch Deployment/target_name.json --output_dir Deployment/compiled_results_nano --net_name "yolov5_nano"
```