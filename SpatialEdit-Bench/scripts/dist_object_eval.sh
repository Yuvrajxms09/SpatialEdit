RESOLUTION=1024

META_FILE=/pfs/yichengxiao/data/space_edit/SpatialEdit_Bench_Meta_File.json
SAVE=/pfs/yichengxiao/projects/SpatialEdit_github/SpatialEdit-Bench/eval_output/spatialedit
BENCH_DATA_DIR=/pfs/yichengxiao/data/space_edit/SpatialEdit_Bench_Data

# run benchmark
cd SpatialEdit-Bench/object_level_eval

EDITED_IMAGES_DIR=$(dirname "$SAVE")
MODEL_NAME=$(basename "$SAVE")
BENCH_SAVE_DIR=$EDITED_IMAGES_DIR/spatialedit_bench_results
LANGUAGE=en
# BACKBONE=gpt4o
# BACKBONE=gemini3_pro
# BACKBONE=gemini3flash
BACKBONE=gemini2flash

python3 calculate_score.py --model_name "$MODEL_NAME" \
  --save_dir "$BENCH_SAVE_DIR" \
  --backbone "$BACKBONE" \
  --edited_images_dir "$EDITED_IMAGES_DIR" \
  --metadata_path "$META_FILE" \
  --bench-data-dir="$BENCH_DATA_DIR" \
  --instruction_language "$LANGUAGE"

python3 calculate_statistics.py --model_name "$MODEL_NAME" \
  --save_path "$BENCH_SAVE_DIR" \
  --backbone "$BACKBONE" \
  --language "$LANGUAGE"

echo "Benchmark Done."
