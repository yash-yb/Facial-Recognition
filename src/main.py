import "Landmarks/capture_frame.py"

choose=0

def main():
    global choose
    while True:
        print("1. Capture new face")
        print("2. Detect face")
        print("3. Capture face")
        print("4. Exit")
        choose = int(input("Enter your choice: "))
        if choose == 1:
            new_face()
        elif choose == 2:
            detect_face()
        elif choose == 3:
            capture_face()
        elif choose == 4:
            break
        else:
            print("Invalid choice. Please try again.")