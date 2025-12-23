"""
Hand Detection Module using MediaPipe
Detects hands and recognizes basic gestures
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Detects hands and recognizes gestures"""
    
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialize hand detector
        
        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        
    def find_hands(self, img, draw=True):
        """
        Find hands in image
        
        Args:
            img: Input image (BGR format)
            draw: Whether to draw hand landmarks
            
        Returns:
            Image with hand landmarks drawn (if draw=True)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return img
    
    def find_position(self, img, hand_no=0):
        """
        Get positions of all hand landmarks
        
        Args:
            img: Input image
            hand_no: Which hand to get landmarks from (0 = first hand)
            
        Returns:
            List of landmarks [id, x, y]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                h, w, c = img.shape
                for id, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """
        Count how many fingers are up
        
        Args:
            landmark_list: List of hand landmarks
            
        Returns:
            List of 5 values (1 if finger is up, 0 if down) [thumb, index, middle, ring, pinky]
        """
        if len(landmark_list) == 0:
            return [0, 0, 0, 0, 0]
        
        fingers = []
        
        # Thumb (check if tip is to the left/right of IP joint)
        if landmark_list[4][1] < landmark_list[3][1]:  # Thumb tip left of IP
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other 4 fingers
        tip_ids = [8, 12, 16, 20]
        for tip_id in tip_ids:
            if landmark_list[tip_id][2] < landmark_list[tip_id - 2][2]:  # Tip above PIP joint
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_gesture(self, landmark_list):
        """
        Recognize hand gesture
        
        Args:
            landmark_list: List of hand landmarks
            
        Returns:
            String describing the gesture
        """
        if len(landmark_list) == 0:
            return "none"
        
        fingers = self.fingers_up(landmark_list)
        total_fingers = fingers.count(1)
        
        # Gesture recognition
        if total_fingers == 0:
            return "fist"
        elif total_fingers == 1:
            if fingers[1] == 1:  # Only index finger
                return "one"
            elif fingers[0] == 1:  # Only thumb
                return "thumbs_up"
        elif total_fingers == 2:
            if fingers[1] == 1 and fingers[2] == 1:  # Index and middle
                return "two"
            elif fingers[0] == 1 and fingers[1] == 1:  # Thumb and index (pinch)
                return "pinch"
        elif total_fingers == 3:
            return "three"
        elif total_fingers == 4:
            return "four"
        elif total_fingers == 5:
            return "open"
        
        return f"{total_fingers}_fingers"
    
    def get_hand_center(self, landmark_list):
        """
        Get center point of hand (wrist position)
        
        Args:
            landmark_list: List of hand landmarks
            
        Returns:
            [x, y] coordinates of hand center
        """
        if len(landmark_list) == 0:
            return None
        return [landmark_list[0][1], landmark_list[0][2]]
    
    def get_pinch_distance(self, landmark_list):
        """
        Get distance between thumb and index finger tips
        
        Args:
            landmark_list: List of hand landmarks
            
        Returns:
            Distance in pixels
        """
        if len(landmark_list) < 21:
            return 0
        
        thumb_tip = landmark_list[4]
        index_tip = landmark_list[8]
        
        distance = np.sqrt((thumb_tip[1] - index_tip[1])**2 + 
                          (thumb_tip[2] - index_tip[2])**2)
        return distance


def main():
    """Test the hand detector"""
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    print("=" * 50)
    print("Hand Detector Test")
    print("=" * 50)
    print("Show different gestures:")
    print("  ✊ Fist")
    print("  ☝️  One finger")
    print("  ✌️  Two fingers")
    print("  ✋ Open hand")
    print("\nPress 'Q' to quit")
    print("=" * 50)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Flip for selfie view
        img = cv2.flip(img, 1)
        
        # Detect hands
        img = detector.find_hands(img)
        landmarks = detector.find_position(img)
        
        # Get gesture
        if landmarks:
            gesture = detector.get_gesture(landmarks)
            fingers = detector.fingers_up(landmarks)
            hand_center = detector.get_hand_center(landmarks)
            
            # Display info
            cv2.putText(img, f"Gesture: {gesture}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Fingers: {fingers}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw center point
            if hand_center:
                cv2.circle(img, tuple(hand_center), 10, (0, 0, 255), cv2.FILLED)
        
        # Display
        cv2.imshow("Hand Detector Test", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
