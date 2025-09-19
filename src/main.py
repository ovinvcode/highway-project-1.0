# src/main.py
import cv2
import time

# Import our updated analysis module
import camera_feed
import hardware_control # We'll keep this for the ticket printing simulation

def run_system():
    cap = cv2.VideoCapture("1234.webp")
    print("System started. Looking for vehicles...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # This function now only detects YOLO objects
        detected_objects, display_frame = camera_feed.analyze_frame(frame)

        # --- YOUR SIMPLIFIED RULES ENGINE ---
        is_vehicle_present = "car" in detected_objects or "truck" in detected_objects or "bus" in detected_objects

        if is_vehicle_present:
            print(f"Vehicle Detected! Type: {detected_objects}")
            
            # Simulate printing a ticket
            hardware_control.print_ticket("Highway Entrance", time.ctime())
            
            # Wait for a few seconds before looking again to avoid multiple triggers
            time.sleep(5)
        
        cv2.imshow("Vehicle Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()