# Importamos librerias
from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np
from tkinter import filedialog
from tkinter import font
import mediapipe as mp

################################################################################################
def cambiofondo():
    mp_selfie_segmentation = mp.solutions.selfie_segmentation 

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    BG_COLOR =(219, 203, 255)
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:

        while True:
            
            ret, frame = cap.read()
            if ret == False:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(frame_rgb)

            _, th =cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
        
            th = th.astype(np.uint8)
            th_inv = cv2.bitwise_not(th)
        # cv2.imshow("results.segmentation_mask",results.segmentation_mask)
        
        #background
            bg_image =np.ones(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            bg =cv2.bitwise_and(bg_image, bg_image, mask=th_inv)

        #foreground
            fg =cv2.bitwise_and(frame, frame, mask=th)

        #background +foreground
            output_image = cv2.add(bg, fg)

            #cv2.imshow("th",th)      
            #cv2.imshow("th_inv",th_inv) 
            #cv2.imshow("fg",fg) 
           # cv2.imshow("bg",bg) 
            cv2.imshow("output_image",output_image)
           # cv2.imshow("Frame",frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
imagen_actual = None
# Función para importar imagen

def importar_imagen():
    global imagen_actual
    filename = filedialog.askopenfilename(initialdir="/", title="Seleccionar imagen", filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"),))
    if filename:
        imagen_actual = cv2.imread(filename)
        imagen_actual = cv2.cvtColor(imagen_actual, cv2.COLOR_BGR2RGB)
        imagen_actual = imutils.resize(imagen_actual, width=640)
        frame_img = imagen_actual.copy()
        im = Image.fromarray(imagen_actual)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        cv2.destroyAllWindows()


###############################################################################################################3
# Function to update the image based on the slider values
# Colores
def update_image():
    global slider1, slider11, slider2, slider22, slider3, slider33, img_edit, img_tk

    # Extraemos el valor de los sliders
    r = slider1.get()
    g = slider2.get()
    b = slider3.get()
    r1 = slider11.get()
    g1 = slider22.get()
    b1 = slider33.get()

    imagen_rgb = imagen_actual.copy()
    imagen_rgb[:, :, 0] = np.clip(imagen_rgb[:, :, 0] + r, r1, 255)
    imagen_rgb[:, :, 1] = np.clip(imagen_rgb[:, :, 1] + g, g1, 255)
    imagen_rgb[:, :, 2] = np.clip(imagen_rgb[:, :, 2] + b, b1, 255)
    img_tk = ImageTk.PhotoImage(Image.fromarray(imagen_rgb))
    img_edit = imagen_rgb.copy()

    
    #Mostrar imagen editada
    lblVideo.configure(image=img_tk)
    lblVideo.image = img_tk



# Función para guardar imagen
def guardar_imagen():
    global img_edit_final
    filename = filedialog.asksaveasfilename(initialdir="/", title="Guardar imagen", filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"),))
    if filename:
        # Convertir la imagen a formato BGR
        img_edit_final = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
        # Guardar la imagen editada
        cv2.imwrite(filename, img_edit_final)



###############################################################################################################3

# Funcion Visualizar
def visualizar():
    global frame_img, pantalla, frame, rgb, hsv, gray, slival1, slival11, slival2, slival22, slival3, slival33, img_edit_final, img_tk
    # Leemos la videocaptura
    if cap is not None:
        ret, frame = cap.read()

        # Si es correcta
        if ret == True:

            if (rgb == 1 and hsv == 0 and gray == 0):
                # Color BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elif rgb == 0 and hsv == 1 and gray == 0:
                # Color HSV
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                

            elif (rgb == 0 and hsv == 0 and gray == 1):
                # Color GRAY
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



            # Rendimensionamos el video
            frame = imutils.resize(frame, width=640)

            # Convertimos el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)

        else:
            cap.release()

# Conversion de color
def hsvf():
    global hsv, rgb, gray, imagen_actual, img_edit_final
    hsv = 1
    rgb = 0
    gray = 0
    if imagen_actual is not None:
        frame = imagen_actual.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        img_edit_final = frame
        frame = imutils.resize(frame, width=640)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        
# RGB
def rgbf():
    global hsv, rgb, gray, imagen_actual
    hsv = 0
    rgb = 1
    gray = 0
    if imagen_actual is not None:
        frame = imagen_actual.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = imutils.resize(frame, width=640)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

# GRAY
def grayf():
    global hsv, rgb, gray, imagen_actual
    hsv = 0
    rgb = 0
    gray = 1
    if imagen_actual is not None:
        frame = imagen_actual.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = imutils.resize(frame, width=640)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

# Colores
def colores():
    global slider1, slider11, slider2, slider22, slider33, detcolor
    global slival1, slival11, slival2, slival22, slival3, slival33, slival4, slival44

    # Activamos deteccion de color
    detcolor = 1

    # Extraemos el sliders H
    slival1 = slider1.get()
    slival11 = slider11.get()
    print(slival1, slival11)
    # Extraemos el sliders S
    slival2 = slider2.get()
    slival22 = slider22.get()
    print(slival2, slival22)
    # Extraemos el sliders V
    slival3 = slider3.get()
    slival33 = slider33.get()
    print(slival3, slival33)

    # Deteccion de color
    if detcolor == 1:
        # Deteccion de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Establecemos el rango minimo y maximo para la codificacion HSV
        color_oscuro = np.array([slival1, slival2, slival3])
        color_brilla = np.array([slival11, slival22, slival33])

        # Detectamos los pixeles que esten dentro de los rangos
        mascara = cv2.inRange(hsv, color_oscuro, color_brilla)

        # Mascara
        mask = cv2.bitwise_and(frame, frame, mask=mascara)

        mask = imutils.resize(mask, width=360)

        # Convertimos el video
        im2 = Image.fromarray(mask)
        img2 = ImageTk.PhotoImage(image=im2)

        # Mostramos en el GUI
        lblVideo2.configure(image=img2)
        lblVideo2.image = img2
        lblVideo2.after(10, colores)



# Funcion iniciar
def iniciar():
    global cap
    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizar()
    print("GRABANDO")

# Funcion finalizar
def finalizar():
    cap.release()
    cv2.destroyAllWindows()
    print("FIN")

# Variables
cap = None
hsv = 0
gray = 0
rgb = 1
detcolor = 0


#  Ventana Principal
# Pantalla
pantalla = Tk()
pantalla.title("GUI | TKINTER | VISION ARTIFICIAL | PEPITO PEREZ")
pantalla.geometry("1280x720")  # Asignamos la dimension de la ventana
fuente = font.Font(weight="bold")
# Fondo
imagenF = PhotoImage(file="fondo.png")
background = Label(image = imagenF, text = "Fondito")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

# Interfaz
texto1 = Label(pantalla, text="VIDEO EN TIEMPO REAL: ",bg="#8abdb0",fg="white")
texto1.configure(font=fuente)
texto1.place(x = 530, y = 10)

texto2 = Label(pantalla, text="IMPORTAR Y GUARDAR: ", bg="#e5aa5c", fg="white")
texto2.configure(font=fuente)
texto2.place(x = 1025, y = 50)

texto3 = Label(pantalla, text="DETECCION DE COLOR: ", bg="#614d56", fg="white")
texto3.configure(font=fuente)
texto3.place(x = 110, y = 100)

texto4 = Label(pantalla, text= ": ", bg="#e3a85a", fg="white")
texto4.configure(font=fuente)
texto2.place(x = 1025, y = 325)



# Botones
fuente = font.Font(weight="bold")
# Iniciar Video

inicio = Button(pantalla, text="Iniciar", height="1", width="18", command=iniciar,bg="#83b5a9",fg="white",font=fuente)
inicio.place(x = 82, y = 480)

# Finalizar Video

fin = Button(pantalla, text="Finalizar", height="1", width="18", command=finalizar,bg="#e5deb2",fg="white",font=fuente)
fin.place(x = 82, y = 580)

# HSV

bhsv = Button(pantalla, text="HSV", height="1", width="18", command=hsvf, bg="#e4ddb1",fg="white",font=fuente)
bhsv.place(x = 1000, y = 150)
# RGB

brgb = Button(pantalla, text="RGB", height="1", width="18", command=rgbf, bg="#e4ddb1",fg="white",font=fuente)
brgb.place(x = 1000, y = 200)
# GRAY

grayb = Button(pantalla, text="GRIS", height="1", width="18", command=grayf, bg="#e6ab5d",fg="white",font=fuente)
grayb.place(x = 1000, y = 250)

# Colores

color = Button(pantalla, text="Colores", height="1", width="18", command=colores,bg="#624e57",fg="white",font=fuente)
color.place(x = 82, y = 350)
#Boton Importar Imagen

importar = Button(pantalla, text="Importar Imagen", height="1", width="18", command=importar_imagen,bg="#e6ab5d",fg="white",font=fuente)
importar.place(x = 1000, y = 360)

# GUARDAR IMAGEN

guardar = Button(pantalla, text="Guardar Imagen",  height="1", width="18", command=guardar_imagen,bg="#e6ab5d",fg="white",font=fuente)
guardar.place(x = 1000, y = 420)

#boton
"""boton = Button(pantalla, text="boton pq si",height="1", width="6", command=update_image)
boton.place(x=82, y=300)"""

#cambiar fondo de video
cambiofn = Button(pantalla, text="cambiar fondo de video", height="1", width="18", command=cambiofondo, bg="#e45a58",fg="white", font=fuente)
cambiofn.place(x = 1000, y = 600)


# Sliders
# Color H
slider1 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider1.place(x = 80, y = 150)
slider11 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider11.place(x = 190, y = 150)
# Color S
slider2 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider2.place(x = 80, y = 200)
slider22 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider22.place(x = 190, y = 200)
# Color V
slider3 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider3.place(x = 80, y = 250)
slider33 = Scale(pantalla, from_ = 0, to = 255, orient=HORIZONTAL, troughcolor="#604c55",bg="#604c55")
slider33.place(x = 190, y = 250)


# Video
lblVideo = Label(pantalla)
lblVideo.place(x = 320, y = 50)

lblVideo2 = Label(pantalla)
lblVideo2.place(x = 470, y = 500)

pantalla.mainloop()