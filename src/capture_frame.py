def capture_single_frame():

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None

    cap.release()
    return frame