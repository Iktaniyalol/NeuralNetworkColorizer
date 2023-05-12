import os
from tkinter import Tk, messagebox, filedialog, Canvas, Button, CENTER, BOTTOM, Label, S
import tkinter.font as font
import keras
from skimage import color
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from keras.optimizers import Adam

directory = os.getcwd()
font_dict = {'fontsize': 6, 'fontweight': 'bold'}
model_file_type = [('Hierarchical Data Format, H5', '.h5')]
pictures_file_type = [("Картинки", ".jpg .jpeg .png .jpe .bmp"), ('JPG', '.jpg'), ('JPEG', '.jpeg'),
                      ('PNG', '.png'), ('JPE', '.jpe'), ('BMP', '.bmp')]
MODEL_PATH = os.path.join(directory, "model.h5")
image_width = 256
image_height = 256


def normalize_l(in_l):
    return (in_l - 50) / 100


def unnormalize_l(in_l):
    return in_l * 100 + 50


def normalize_ab(in_ab):
    return in_ab / 110


def unnormalize_ab(in_ab):
    return in_ab * 110


def select():
    global draw_button, label_selected
    answer = filedialog.askopenfilename(
        parent=window,
        initialdir=directory,
        title="Выберите картинку для раскрашивания:",
        filetypes=pictures_file_type
    )
    if not answer:
        return
    label_selected = Label(master=window, text="Выбрано: " + answer)
    label_selected.pack(pady=(0, 80))
    draw_button = Button(master=window, command=lambda: colorize(answer), text="Раскрасить")
    draw_button['font'] = button_font
    draw_button.pack(pady=(0, 80))


def reset_all():
    global back_button, canvas
    back_button.destroy()
    canvas.destroy()
    toolbar.destroy()
    def_button()


def def_button():
    global select_button
    select_button = Button(master=window, command=select, text="Выбрать изображение")
    select_button['font'] = button_font
    select_button.pack(pady=(100, 20))


def colorize(path):
    global output, plot_figure, select_button, draw_button, canvas, back_button, toolbar
    back_button = Button(master=window, command=reset_all, text="Другое изображение")
    back_button['font'] = button_font
    back_button.pack(pady=(0, 0))
    canvas = Canvas(window, width=800, height=500, bg='white')
    canvas.pack()
    label_selected.destroy()
    draw_button.destroy()
    select_button.destroy()
    image = Image.open(path)
    orig_x, orig_y = image.size
    image = image.resize((image_width, image_height)).convert("RGB")
    lab = color.rgb2lab(image)
    img_arr = np.array([normalize_l(lab)[:, :, 0]])
    img_arr = np.expand_dims(img_arr, axis=-1)

    test_img_unnorm = unnormalize_l(img_arr)
    grayscale = np.zeros((image_width, image_height, 3))
    grayscale[:, :, 0] = test_img_unnorm[0][:, :, 0]
    gray_img = Image.fromarray(np.uint8(color.lab2rgb(grayscale) * 255))

    output1 = model.predict(img_arr)
    output1 = unnormalize_ab(output1)
    result = np.zeros((image_width, image_height, 3))
    result[:, :, 0] = test_img_unnorm[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    color_img = Image.fromarray(np.uint8(color.lab2rgb(result) * 255))
    plot_figure = Figure(figsize=(6, 3), dpi=300)
    plot1 = plot_figure.add_subplot(1, 2, 1)
    plot1.imshow(gray_img.resize((orig_x, orig_y)))
    plot1.set_title('Оригинал', font_dict)
    plot1.axis('off')
    plot1.set_aspect('equal')

    plot2 = plot_figure.add_subplot(1, 2, 2)
    plot2.imshow(color_img.resize((orig_x, orig_y)))
    plot2.set_title('Результат', font_dict)
    plot2.axis('off')
    plot2.set_aspect('equal')

    output = FigureCanvasTkAgg(plot_figure, master=canvas)
    output.draw()
    output.get_tk_widget().pack(expand=True)

    toolbar = NavigationToolbar2Tk(output, window)
    toolbar.update()
    output.get_tk_widget().pack()


window = Tk()
window.title("Раскрашивание чёрно-белых изображений с помощью нейронной сети")
window.geometry('1920x1080')
window.minsize(1080, 720)
window['bg'] = "white"

toolbar = None
canvas = None
output = None
plot_figure = None
back_button = None
draw_button = None
select_button = None
label_selected = None
button_font = font.Font(family='Comic Sans', size=25, weight="bold")
def_button()

if not os.path.isfile(MODEL_PATH):
    messagebox.showinfo("Ошибка", "Модель нейронной сети не найдена. Убедитесь, что файл model.h5 лежит в одной "
                                  "директории с файлом запуска приложения.")
    exit()
else:
    model = keras.models.load_model(MODEL_PATH)
    adamOpti = Adam(learning_rate=0.0001)
    model.compile(optimizer=adamOpti, loss="mse", metrics=["accuracy"])

window.mainloop()
