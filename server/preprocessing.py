import numpy as np

def extract_landmarks(multi_hand_landmarks) -> list:
    """
    Extracts, normalizes, and pads or slices MediaPipe hand landmarks to 126 features.
    This logic matches the exact training and local inference preprocessing.
    """
    hands_data = []
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            temp_hand = []
            max_val = 0
            for lm in hand_landmarks.landmark:
                rx = lm.x - base_x
                ry = lm.y - base_y
                rz = lm.z - base_z
                temp_hand.extend([rx, ry, rz])
                max_val = max(max_val, abs(rx), abs(ry), abs(rz))
            
            if max_val > 0:
                temp_hand = [val / max_val for val in temp_hand]
            hands_data.extend(temp_hand)
            
        if len(multi_hand_landmarks) == 1:
            hands_data.extend([0.0] * 63)
        hands_data = hands_data[:126]
    else:
        hands_data = [0.0] * 126
        
    return hands_data
