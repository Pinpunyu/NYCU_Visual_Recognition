set -e                      

DATA_ROOT="hw3_data_release"
OUT_DIR="outputs/v2_SGD_MultiStepLR_epoch100_3"            
EPOCHS=100
BATCH_SIZE=2
# LR=5e-5
LR=0.005
WORKERS=2
SAVE_JSON="outputs/v2_SGD_MultiStepLR_epoch100_3/test_results.json"
IOU_THRESH=0.8

echo "==> Training..."
python main.py train --data-root "$DATA_ROOT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --out-dir "$OUT_DIR" --workers "$WORKERS" --iou-thresh "$IOU_THRESH" 

BEST_WEIGHTS="$OUT_DIR/model_best.pth"
echo "==> Predicting with $BEST_WEIGHTS ..."
python main.py predict --data-root "$DATA_ROOT" --weights "$BEST_WEIGHTS" --save-path "$SAVE_JSON" --out-dir "$OUT_DIR" --workers "$WORKERS" --iou-thresh "$IOU_THRESH" 

echo "All done! Results saved to $SAVE_JSON"
