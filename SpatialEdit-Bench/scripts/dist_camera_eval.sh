
VGGT=your_base_path/VGGT-1B/model.pt
YOLO=your_base_path/yolo/yolo26x.pt

EVAL_DATA='{
  "model_name_A":"path to results of model A",
  "model_name_B":"path to results of model B"
}'

VIEWPOINT_ERROR_RESULT=SpatialEdit-Bench/camera_result/viewpoint_error.csv
FRAMING_ERROR_RESULT=SpatialEdit-Bench/camera_result/framing_error
META_DATA_FILE=your_base_path/SpatialEdit_Bench_Meta_File.json

export cuda_visible_devices=0,1,2,3
torchrun --nproc_per_node=4 SpatialEdit-Bench/camera_level_eval/VE_metric_ddp.py \
  --models_json "$EVAL_DATA" \
  --vggt_ckpt "$VGGT" \
  --meta_data_file "$META_DATA_FILE" \
  --out_csv "$VIEWPOINT_ERROR_RESULT"

echo "viewpoint error calculate Done."

torchrun --nproc_per_node=4 --master-port=12345 SpatialEdit-Bench/camera_level_eval/FE_metric_ddp.py \
  --yolo_model $YOLO \
  --datasets_json "$EVAL_DATA" \
  --meta_data_file "$META_DATA_FILE" \
  --out_dir "$FRAMING_ERROR_RESULT"

echo "framing error calculate Done."
