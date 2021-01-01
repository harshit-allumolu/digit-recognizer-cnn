"""
	Author : Harshit Allumolu <allumoluharshit@gmail.com>
	National Institute of Technology, Warangal.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # to avoid tf warnings
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageGrab
import numpy as np
from tkinter import *
import tkinter as tk

model = load_model("digit.pb")

# predict function
def predict(image):
    image = image.resize((28,28))
    image = image.convert('L')
    image = np.invert(np.array(image))
    image = image.reshape(1,28,28,1)
    prediction = model.predict(image)[0]
    return np.argmax(prediction)


class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        # widgets
        self.canvas = tk.Canvas(self,width=500,height=500,bg='white',cursor='cross')
        self.label1 = tk.Label(self,text = 'Draw a digit here',font=('Ubuntu',20))
        self.label2 = tk.Label(self,font=('Ubuntu',20))
        self.frame = tk.Frame(self,width=500)
        clear_button = tk.Button(self.frame,text="Clear",font=('Ubuntu',15),activebackground='green',command=self.clear)  
        digit_button = tk.Button(self.frame,text="Recognise",font=('Ubuntu',15),activebackground='green',command=self.digit)

        # grid
        self.canvas.grid(row=1,column=0,pady=2,padx=2,sticky=W)
        self.label1.grid(row=0,column=0,pady=2,padx=2,sticky=W)
        self.label2.grid(row=3,column=0,pady=2,padx=2,sticky=W)
        self.frame.grid(row=2,column=0,pady=2,padx=2)
        
        clear_button.pack(side="left")
        digit_button.pack(side="right")

        self.canvas.bind("<B1-Motion>",self.draw)

    def clear(self):
        self.canvas.delete("all")
        self.label2.configure(text = "")
    
    def digit(self):
        a = self.winfo_rootx() + self.canvas.winfo_x()
        b = self.winfo_rooty() + self.canvas.winfo_y()
        c = a + self.canvas.winfo_width()
        d = b + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((a,b,c,d))
        #image.save("image.png")
        predicted_digit = predict(image)
        self.label2.configure(text = "Digit recognised as : {}".format(str(predicted_digit)))
    
    def draw(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    

myapp = App()
myapp.title("Harshit's Digit Recognizer")
myapp.mainloop()
