

import numpy as np
import cv2 as cv

camTest = 0
if camTest:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Naming a window and giving it right size
        cv.namedWindow("Resize", cv.WINDOW_NORMAL)
        cv.setWindowProperty("Resize", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        #cv.resizeWindow("Resize", 700, 200)            # Using resizeWindow()

        cv.imshow('Resize', frame)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


strTest = 1
if strTest:
    class MyClass:
        x = 0
        y = ""

        def __init__(self, anyNumber, anyString):
            self.x = anyNumber
            self.y = anyString
        def __str__ (self):
            return 'MyClass(x=' + str(self.x) + ' ,y=' + self.y + ')'
        
    

    myObject = MyClass(12345, "Hello")

    #print(myObject.__str__())
    print("myObject: ", myObject)
    #print(str(myObject))
    print(myObject.__repr__())



