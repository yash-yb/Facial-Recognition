import capture_frame as cf
def main():
    choose=0
    print("Welcome to the Facial Access System")
    while True:
        print("1. Capture new face")
        print("2. Detect face")
        print("3. Capture face")
        print("4. Exit")
        choose = int(input("Enter your choice: "))
        if choose == 1:
            cf.new_face()
        elif choose == 2:
            cf.detect_face()
        elif choose == 3:
            cf.capture_face()
        elif choose == 4:
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()