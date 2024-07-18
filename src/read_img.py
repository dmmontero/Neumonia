import pydicom
import numpy as np
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk


class Imagen:
    @classmethod
    def read_dicom_file(self, path):
        img = pydicom.read_file(path)
        img_array = img.pixel_array
        img2show = Image.fromarray(img_array)
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        return img_RGB, img2show

    @classmethod
    def read_jpg_file(path):
        img = cv2.imread(path)
        img_array = np.asarray(img)
        img2show = Image.fromarray(img_array)
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        return img2, img2show

    @classmethod
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            array, img2show = self.read_dicom_file(filepath)
            img1 = img2show.resize((250, 250), Image.LANCZOS)
            img1 = ImageTk.PhotoImage(self.img1)
            return array, img1
