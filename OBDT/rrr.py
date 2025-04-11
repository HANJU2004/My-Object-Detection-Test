import tkinter as tk
from tkinter import Label
import cv2
import torch
from PIL import Image, ImageTk
from model import MyNet
from config import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class CameraApp:
    def __init__(self, window, window_title, video_source=0,detect=True):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Open the video source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Set video source width and height
        self.vid.set(3, 448)
        self.vid.set(4, 448)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=448, height=448)
        self.canvas.pack()

        # Create a label to display mouse coordinates
        self.coord_label = tk.Label(window, bg='black', fg='yellow', font=('Arial', 12))
        self.coord_label.place(x=10, y=10)

        # Bind the mouse motion event to the callback function
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # self.net=MyNet().to(device)
        # self.net.load_state_dict(torch.load("weights/target_recognition_step_250000.pth"))
        # self.net.eval()
        # self.detect=detect

        # Update the canvas with the new frame
        self.update()
        self.window.mainloop()

    def update(self):
        self.canvas.delete("all")
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            if self.detect==True:
                with torch.no_grad():
                    output_list=self.net.forward_ret_pxywhcls_list(frame)

                for output_list_classified in output_list:
                    for box in output_list_classified:
                        if box !=[]:
                            self.canvas.create_rectangle(image_size*box[1], image_size*box[2], image_size*box[3], image_size*box[4], outline="yellow",width=3)


        # After 15 milliseconds, call update again
        self.window.after(15, self.update)

    def on_mouse_move(self, event):
        # Get mouse coordinates relative to the canvas
        x, y = event.x, event.y
        coord_text = f"Mouse coordinates: x={x}, y={y}"
        self.coord_label.config(text=coord_text)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__=="__main__":
    # Create a window and pass it to the CameraApp class
    window = tk.Tk()
    app = CameraApp(window, "Camera App")
