import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, IntVar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os

# Global variables
img = gray_img = processed_img = None

# Main GUI window
root = tk.Tk()
root.title("Image Preprocessing App")
root.geometry("900x750")

# Variables to store user selections for preprocessing
apply_hist_eq, apply_gaussian_blur, apply_canny_edge = IntVar(), IntVar(), IntVar()

# Function to load image
def load_image():
    global img, gray_img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        display_image(gray_img, original_img_label)
        calculate_histogram(gray_img, hist_original_label)

# Function to display image on GUI
def display_image(img, label):
    img = Image.fromarray(img).resize((300, 300))
    imgtk = ImageTk.PhotoImage(image=img)
    label.config(image=imgtk)
    label.image = imgtk

# Function to apply selected preprocessing techniques
def process_image():
    global processed_img
    if gray_img is None:
        return

    processed_img = gray_img.copy()
    if apply_hist_eq.get() == 1:
        processed_img = cv2.equalizeHist(processed_img)
    if apply_gaussian_blur.get() == 1:
        processed_img = cv2.GaussianBlur(processed_img, (9, 9), 2)
    if apply_canny_edge.get() == 1:
        processed_img = cv2.Canny(processed_img, 50, 150)

    display_image(processed_img, processed_img_label)
    calculate_histogram(processed_img, hist_equalized_label, equalized=True)
    calculate_quality_metrics(gray_img, processed_img)

    filters = [
        "Histogram Equalization" if apply_hist_eq.get() else "",
        "Gaussian Blur" if apply_gaussian_blur.get() else "",
        "Canny Edge Detection" if apply_canny_edge.get() else ""
    ]
    processed_img_label_text.config(text="Processed Image with " + ", ".join(filter(None, filters)))

# Function to calculate and plot histogram
def calculate_histogram(image, label, equalized=False):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(hist)
    plt.subplots_adjust(bottom=0.30)
    hist_filename = "hist_equalized.png" if equalized else "hist_original.png"
    plt.savefig(hist_filename)
    plt.close()
    hist_image = cv2.imread(hist_filename)
    display_image(cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB), label)
    os.remove(hist_filename)

# Function to calculate PSNR and MSE
def calculate_quality_metrics(original, processed):
    mse = np.mean((original - processed) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    """psnr_label.config(text=f"PSNR: {psnr:.2f} dB")
    mse_label.config(text=f"MSE: {mse:.2f}")"""

# GUI Layout
frame_top = tk.Frame(root)
frame_top.pack()
tk.Button(frame_top, text="Select Image", command=load_image, width=20, height=2).grid(row=0, column=0, padx=10, pady=10)
tk.Button(frame_top, text="Process Image", command=process_image, width=20, height=2).grid(row=0, column=1, padx=10, pady=10)
tk.Button(frame_top, text="Exit", command=root.destroy, width=20, height=2).grid(row=0, column=2, padx=10, pady=10)

# Checkboxes
tk.Checkbutton(frame_top, text="Histogram Equalization", variable=apply_hist_eq).grid(row=1, column=0, padx=5, pady=5)
tk.Checkbutton(frame_top, text="Gaussian Blur", variable=apply_gaussian_blur).grid(row=1, column=1, padx=5, pady=5)
tk.Checkbutton(frame_top, text="Canny Edge Detection", variable=apply_canny_edge).grid(row=1, column=2, padx=5, pady=5)

# Image Display Section
frame_images = tk.Frame(root)
frame_images.pack()
original_img_label = tk.Label(frame_images)
original_img_label.grid(row=0, column=0, padx=10, pady=10)
tk.Label(frame_images, text="Original Image").grid(row=1, column=0)
processed_img_label = tk.Label(frame_images)
processed_img_label.grid(row=0, column=1, padx=10, pady=10)
processed_img_label_text = tk.Label(frame_images, text="Processed Image")
processed_img_label_text.grid(row=1, column=1)

# Histogram Display Section
frame_hist = tk.Frame(root)
frame_hist.pack()
hist_original_label = tk.Label(frame_hist)
hist_original_label.grid(row=0, column=0, padx=10, pady=10)
tk.Label(frame_hist, text="Original Histogram").grid(row=1, column=0)
hist_equalized_label = tk.Label(frame_hist)
hist_equalized_label.grid(row=0, column=1, padx=10, pady=10)
tk.Label(frame_hist, text="Processed Histogram").grid(row=1, column=1)

root.mainloop()
