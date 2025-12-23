"""
Hand Gesture 3D Controller - Enhanced Version
Control a realistic 3D cube using hand gestures including pinch
"""

import cv2
import numpy as np
from hand_detector import HandDetector


class Object3D:
    """Enhanced 3D object with realistic shading and lighting"""
    
    def __init__(self, position=[320, 240]):
        """Initialize 3D object"""
        self.rotation_x = 30
        self.rotation_y = 45
        self.rotation_z = 0
        self.scale = 100
        self.position_x = position[0]
        self.position_y = position[1]
        
        # Lighting
        self.light_direction = np.array([1, -1, 1])
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
        
    def rotate(self, dx, dy, dz):
        """Rotate object"""
        self.rotation_x += dx
        self.rotation_y += dy
        self.rotation_z += dz
        
    def move(self, x, y):
        """Move object"""
        self.position_x = x
        self.position_y = y
        
    def set_scale(self, scale):
        """Set scale (50-200)"""
        self.scale = max(50, min(200, scale))
    
    def calculate_lighting(self, normal):
        """Calculate lighting intensity for a surface"""
        # Diffuse lighting
        intensity = max(0, np.dot(normal, self.light_direction))
        # Add ambient light
        intensity = 0.3 + 0.7 * intensity
        return intensity
    
    def draw_cube(self, img):
        """Draw a realistic 3D cube with lighting"""
        # Define cube vertices
        cube_points = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
        ], dtype=np.float32)
        
        # Create rotation matrices
        rx = np.radians(self.rotation_x)
        ry = np.radians(self.rotation_y)
        rz = np.radians(self.rotation_z)
        
        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Apply all rotations
        rotated = cube_points @ rot_x @ rot_y @ rot_z
        
        # Project to 2D with perspective
        projected = []
        depths = []
        for point in rotated:
            z = point[2] + 5  # Move away from camera
            factor = 200 / z * self.scale / 100
            x = int(point[0] * factor + self.position_x)
            y = int(point[1] * factor + self.position_y)
            projected.append([x, y])
            depths.append(z)
        
        projected = np.array(projected, dtype=np.int32)
        
        # Define faces with base colors and normals
        faces = [
            # (indices, base_color, normal)
            ([4, 5, 6, 7], (50, 150, 255), np.array([0, 0, 1])),    # Front (Blue-ish)
            ([0, 1, 2, 3], (255, 50, 50), np.array([0, 0, -1])),    # Back (Red)
            ([0, 1, 5, 4], (50, 255, 50), np.array([0, -1, 0])),    # Bottom (Green)
            ([2, 3, 7, 6], (255, 255, 50), np.array([0, 1, 0])),    # Top (Yellow)
            ([0, 3, 7, 4], (255, 50, 255), np.array([-1, 0, 0])),   # Left (Magenta)
            ([1, 2, 6, 5], (50, 255, 255), np.array([1, 0, 0]))     # Right (Cyan)
        ]
        
        # Calculate average depth for each face (for correct drawing order)
        face_depths = []
        for face_indices, _, _ in faces:
            avg_depth = sum(depths[i] for i in face_indices) / len(face_indices)
            face_depths.append(avg_depth)
        
        # Sort faces by depth (back to front)
        sorted_faces = sorted(zip(faces, face_depths), key=lambda x: -x[1])
        
        # Draw faces with lighting
        for (face_indices, base_color, normal), _ in sorted_faces:
            points = np.array([projected[i] for i in face_indices])
            
            # Rotate normal vector
            rotated_normal = normal @ rot_x @ rot_y @ rot_z
            
            # Calculate lighting
            intensity = self.calculate_lighting(rotated_normal)
            
            # Apply lighting to color
            lit_color = tuple(int(c * intensity) for c in base_color)
            
            # Draw filled face
            overlay = img.copy()
            cv2.fillPoly(overlay, [points], lit_color)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Draw edges with darker color
            edge_color = tuple(max(0, int(c * 0.3)) for c in lit_color)
            for i in range(len(points)):
                pt1 = tuple(points[i])
                pt2 = tuple(points[(i + 1) % len(points)])
                cv2.line(img, pt1, pt2, edge_color, 3)
        
        # Add specular highlights on edges
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Back face
            (4,5), (5,6), (6,7), (7,4),  # Front face
            (0,4), (1,5), (2,6), (3,7)   # Connecting edges
        ]
        
        for edge in edges:
            pt1 = tuple(projected[edge[0]])
            pt2 = tuple(projected[edge[1]])
            cv2.line(img, pt1, pt2, (20, 20, 20), 2)
        
        return img


class GestureController:
    """Enhanced controller with pinch gesture support"""
    
    def __init__(self):
        """Initialize controller"""
        self.detector = HandDetector(max_hands=1)
        self.obj3d = Object3D()
        
        # Pinch control
        self.pinch_start_scale = 100
        self.pinch_start_distance = 0
        self.is_pinching = False
        
    def update_object(self, gesture, landmarks):
        """Update 3D object based on gesture"""
        if not landmarks:
            return
        
        hand_center = self.detector.get_hand_center(landmarks)
        
        if gesture == "pinch":
            # PINCH: Precise scale control
            pinch_dist = self.detector.get_pinch_distance(landmarks)
            
            if not self.is_pinching:
                # Start pinching
                self.is_pinching = True
                self.pinch_start_scale = self.obj3d.scale
                self.pinch_start_distance = pinch_dist
            else:
                # Continue pinching - scale based on distance change
                if self.pinch_start_distance > 0:
                    scale_factor = pinch_dist / self.pinch_start_distance
                    new_scale = self.pinch_start_scale * scale_factor
                    self.obj3d.set_scale(new_scale)
        else:
            self.is_pinching = False
            
            if gesture == "fist":
                # FIST: Rotate cube based on hand position
                if hand_center:
                    dx = (hand_center[1] - 240) * 0.3
                    dy = (hand_center[0] - 320) * 0.3
                    self.obj3d.rotate(dx, dy, 0)
                    
            elif gesture == "open":
                # OPEN HAND: Scale cube based on hand height
                if hand_center:
                    scale_factor = (480 - hand_center[1]) / 480
                    new_scale = 50 + scale_factor * 150
                    self.obj3d.set_scale(new_scale)
                    
            elif gesture == "two":
                # TWO FINGERS: Move cube
                if hand_center:
                    self.obj3d.move(hand_center[0], hand_center[1])
                    
            elif gesture == "one":
                # ONE FINGER: Spin on Z axis
                self.obj3d.rotate(0, 0, 5)
                
            elif gesture == "three":
                # THREE FINGERS: Reset
                self.obj3d.rotation_x = 30
                self.obj3d.rotation_y = 45
                self.obj3d.rotation_z = 0
                self.obj3d.scale = 100
                self.obj3d.position_x = 320
                self.obj3d.position_y = 240
                
            elif gesture == "four":
                # FOUR FINGERS: Rotate on X axis
                self.obj3d.rotate(5, 0, 0)
    
    def draw_ui(self, img, gesture, fps):
        """Draw enhanced user interface"""
        h, w = img.shape[:2]
        
        # Semi-transparent background panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Title with gradient effect
        cv2.putText(img, "3D CUBE CONTROLLER", (10, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
        
        # Current gesture with emoji-like indicators
        gesture_colors = {
            "fist": (0, 100, 255),
            "open": (0, 255, 0),
            "one": (255, 200, 0),
            "two": (255, 0, 200),
            "three": (200, 0, 255),
            "four": (255, 128, 0),
            "pinch": (255, 0, 128),
            "none": (128, 128, 128)
        }
        
        color = gesture_colors.get(gesture, (255, 255, 255))
        cv2.putText(img, f"Gesture: {gesture.upper()}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Object properties
        info_text = f"Rot: X={int(self.obj3d.rotation_x)} Y={int(self.obj3d.rotation_y)} Z={int(self.obj3d.rotation_z)}"
        cv2.putText(img, info_text, (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(img, f"Scale: {int(self.obj3d.scale)}%  Pos: ({int(self.obj3d.position_x)}, {int(self.obj3d.position_y)})", 
                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # FPS counter
        cv2.putText(img, f"FPS: {int(fps)}", (w - 130, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Bottom instruction panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        instructions = [
            "Fist=Rotate | Open=Scale | Pinch=PreciseScale | Two=Move | One=Spin | Three=Reset",
            "Press 'Q' to quit | 'R' to reset | '1-4' change light direction"
        ]
        
        y_pos = h - 75
        for instruction in instructions:
            cv2.putText(img, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 30
        
        return img
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        # FPS calculation
        prev_time = 0
        import time
        
        print("=" * 70)
        print("🎮 ENHANCED 3D CUBE CONTROLLER WITH PINCH GESTURE")
        print("=" * 70)
        print("\nGesture Controls:")
        print("  ✊ FIST       - Rotate cube (move hand to control)")
        print("  ✋ OPEN       - Scale cube (up = bigger, down = smaller)")
        print("  🤏 PINCH      - Precise scale (pinch closer = smaller)")
        print("  ✌️  TWO       - Move cube to hand position")
        print("  ☝️  ONE       - Spin cube on Z-axis")
        print("  🤟 THREE     - Reset to default")
        print("  🖖 FOUR      - Rotate on X-axis")
        print("\nKeyboard:")
        print("  Q - Quit")
        print("  R - Reset cube")
        print("=" * 70)
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            img = cv2.flip(img, 1)
            
            # Detect hands
            img = self.detector.find_hands(img, draw=True)
            landmarks = self.detector.find_position(img)
            
            # Get gesture
            gesture = "none"
            if landmarks:
                gesture = self.detector.get_gesture(landmarks)
                self.update_object(gesture, landmarks)
            
            # Draw 3D object
            img = self.obj3d.draw_cube(img)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Draw UI
            img = self.draw_ui(img, gesture, fps)
            
            # Display
            cv2.imshow("Enhanced 3D Hand Controller", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.obj3d.rotation_x = 30
                self.obj3d.rotation_y = 45
                self.obj3d.rotation_z = 0
                self.obj3d.scale = 100
                self.obj3d.position_x = 320
                self.obj3d.position_y = 240
                print("Cube reset!")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = GestureController()
    controller.run()
