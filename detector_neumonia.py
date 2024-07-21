#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from tensorflow import keras as K
import csv

# import time
from tkinter import Tk, filedialog, font, ttk, Text, END, HORIZONTAL, StringVar
from tkinter.messagebox import WARNING, askokcancel, showinfo

# import pyautogui
import tkcap
from PIL import Image, ImageTk

from src.read_img import Imagen
from src.integrator import Integrator


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("900x640")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        # cédula
        self.lab3.place(x=65, y=430)
        # resultado
        self.lab4.place(x=65, y=470)

        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=520)
        self.button2.place(x=70, y=520)
        self.button3.place(x=670, y=520)
        self.button4.place(x=520, y=520)
        self.button6.place(x=370, y=520)
        # cédula
        self.text1.place(x=250, y=470, width=180, height=30)
        # predicción
        self.text2.place(x=250, y=430, width=180, height=30)
        # resultados
        self.text3.place(x=450, y=430, width=180, height=30)

        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        # Progress bar widget
        # self.progress_bar = ttk.Progressbar(
        #     self.root,
        #     orient=HORIZONTAL,
        #     length=500,
        #     mode="indeterminate",
        # )
        # self.progress_bar.place(y=600, x=70)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
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
            self.array, img2show = Imagen.read_dicom_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            self.text2.delete("1.0", "end")
            self.text3.delete("1.0", "end")

            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")
        self.label, self.proba, self.heatmap = Integrator.predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.delete("1.0", "end")
        self.text_img2.image_create(END, image=self.img2)

        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")
        self.button1["state"] = "disabled"

    def save_results_csv(self):
        with open("./reportes/historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "./reportes/Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID, overwrite=True)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"./reportes/Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
