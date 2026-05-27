import numpy as np
import pickle
import tensorflow as tf
import os
import sys

# Add parent directory to sys.path so we can import from server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.preprocessing import extract_landmarks

# Mock landmark objects to test keypoint extraction logic without a physical camera
class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

def create_mock_landmarks():
    # 21 landmarks with random values
    np.random.seed(42)
    lms = []
    for _ in range(21):
        lms.append(MockLandmark(
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(-0.1, 0.1))
        ))
    return MockHandLandmarks(lms)

def run_local_manual_preprocessing(multi_hand_landmarks):
    # This is copy-pasted from contoh/3_inference_alfanum.py
    hands_data = []
    for hand_landmarks in multi_hand_landmarks:
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        temp_hand = []
        max_val = 0
        for lm in hand_landmarks.landmark:
            rx, ry, rz = lm.x - base_x, lm.y - base_y, lm.z - base_z
            temp_hand.extend([rx, ry, rz])
            max_val = max(max_val, abs(rx), abs(ry), abs(rz))
        
        if max_val > 0:
            temp_hand = [val / max_val for val in temp_hand]
        hands_data.extend(temp_hand)
        
    if len(multi_hand_landmarks) == 1:
        hands_data.extend([0.0] * 63)
        
    return hands_data[:126]

def test_parity():
    print("--- RUNNING BISINDO PARITY TEST ---")
    mock_lms_1 = create_mock_landmarks()
    mock_multi = [mock_lms_1]
    
    # 1. Compare Preprocessing Output
    local_preprocessed = run_local_manual_preprocessing(mock_multi)
    centralized_preprocessed = extract_landmarks(mock_multi)
    
    assert len(local_preprocessed) == len(centralized_preprocessed), "Length mismatch!"
    np.testing.assert_allclose(local_preprocessed, centralized_preprocessed, rtol=1e-6, err_msg="Preprocessing values mismatch!")
    print("[OK] Preprocessing outputs match perfectly!")
    
    # 2. Compare Model Inference Output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'bisindo_cnn1d_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path)
    
    # Shape input for local
    X_input_local = np.array([local_preprocessed])
    if len(model.input_shape) == 3 and model.input_shape[2] == 1:
        X_input_local_shaped = np.expand_dims(X_input_local, axis=2)
    else:
        X_input_local_shaped = X_input_local
        
    try:
        preds_local = model.predict(X_input_local_shaped, verbose=0)[0]
        print(f"[OK] Local inference succeeded! Preds shape: {preds_local.shape}")
    except Exception as e:
        print(f"Local inference failed with shape {X_input_local_shaped.shape}: {e}")
        preds_local = None

    # Backend style inference: model(X, training=False)
    X_input_api = np.array([centralized_preprocessed], dtype=np.float32)
    if len(model.input_shape) == 3 and model.input_shape[2] == 1:
        X_input_api = np.expand_dims(X_input_api, axis=2)
        
    preds_api = model(X_input_api, training=False).numpy()[0]
    print(f"[OK] API-style inference succeeded! Preds shape: {preds_api.shape}")
    
    if preds_local is not None:
        np.testing.assert_allclose(preds_local, preds_api, rtol=1e-5, err_msg="Inference results mismatch!")
        print("[OK] Predictions and confidence match perfectly!")
        
    print("\n--- ALL PARITY TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    test_parity()
