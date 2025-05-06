set -e                      

DATA_ROOT="hw3_data_release"
OUT_DIR="outputs/test20"          
WORKERS=2
SAVE_JSON="test_results.json"
IOU_THRESH=0.7

BEST_WEIGHTS="$OUT_DIR/model_final.pth"
echo "==> Predicting with $BEST_WEIGHTS ..."
python main.py predict --data-root "$DATA_ROOT" --weights "$BEST_WEIGHTS" --save-path "$SAVE_JSON" --out-dir "$OUT_DIR" --workers "$WORKERS" --iou-thresh "$IOU_THRESH" 

echo "All done! Results saved to $SAVE_JSON"
