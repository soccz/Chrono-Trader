import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os

from data.preprocessor import get_pump_dataset
from utils.logger import logger

def run(fine_tune: bool = False):
    """Trains the multi-class pump prediction model."""
    model_path = os.path.join("models", "pump_classifier.joblib")
    existing_model_path = None

    if fine_tune:
        logger.info("--- Starting Pump Prediction Model Fine-tuning ---")
        dataset = get_pump_dataset(days=3)
        if os.path.exists(model_path):
            existing_model_path = model_path
            logger.info(f"Found existing model at {model_path} for fine-tuning.")
        else:
            logger.warning("No existing pump model found. Falling back to a full training.")
            fine_tune = False
    
    if not fine_tune:
        logger.info("--- Starting Pump Prediction Model Full Training ---")
        dataset = get_pump_dataset()

    if dataset is None or dataset.empty:
        logger.error("Pump dataset could not be generated. Aborting training.")
        return

    y = dataset['pump_label']
    X = dataset.drop(columns=['pump_label', 'market'])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    logger.info(f"Dataset contains {len(dataset)} samples.")
    logger.info(f"Label distribution:\n{y.value_counts(normalize=True).sort_index()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Training XGBoost multi-class classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',  # Changed for multi-class
        num_class=4,               # We have 4 classes (0, 1, 2, 3)
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # For multi-class, sample weighting is a common way to handle imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model.fit(X_train, y_train, sample_weight=sample_weights, xgb_model=existing_model_path)

    logger.info("Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Use 'weighted' average for multi-class metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

    logger.info("--- Multi-Class Pump Prediction Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy:.4f} (전체 예측 중 정답 비율)")
    logger.info(f"Weighted Precision: {precision:.4f} (가중 평균 정밀도)")
    logger.info(f"Weighted Recall: {recall:.4f} (가중 평균 재현율)")
    logger.info(f"Weighted F1-Score: {f1:.4f} (가중 평균 F1 점수)")
    logger.info(f"Weighted ROC AUC: {roc_auc:.4f} (모델의 전반적인 분류 성능)")
    logger.info("--------------------------------------------------")

    joblib.dump(model, model_path)
    logger.info(f"Pump prediction model saved to {model_path}")

if __name__ == "__main__":
    run()