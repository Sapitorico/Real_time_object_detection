#!/usr/bin/python3
import tkinter as tk
import tkinter.messagebox as mb
top = tk.Tk()
top.geometry("800x500")
top.title("test app")
def camerafunction():
   import numpy as np
   import cv2 as cv
   cap = cv.VideoCapture(0)
   # if not cap.isOpened():
   #    print("Cannot open camera")
   #    exit()
   while True:
      # Capture frame-by-frame
      ret, frame = cap.read()
      frame = cv.flip(frame, 1)
      # if frame is read correctly ret is True
      if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
      # Our operations on the frame come here
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      # Display the resulting frame
      cv.imshow('frame', gray)
      if cv.waitKey(1) == ord('q'):
         break
      # When everything done, release the capture
   cap.release()
   cv.destroyAllWindows()

B = tk.Button(top, text ="Hello", command = camerafunction)
B.pack()
top.mainloop()