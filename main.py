import argparse
import gesture_detection.aruco_detector
import gesture_detection.pinch_detector 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='pinch', 
                        choices=['pinch', 'aruco'], 
                        help='Which method to run? pinch=MediaPipe hand pinch, aruco=ArUco-based pointer.')
    args = parser.parse_args()

    if args.method == 'pinch':
        gesture_detection.pinch_detector.pinch_control_demo()
    elif args.method == 'aruco':
        gesture_detection.aruco_detector.aruco_pinch_demo()

if __name__ == "__main__":
    main()
