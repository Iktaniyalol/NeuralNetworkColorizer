import os
from tkinter import Tk, messagebox, filedialog, Canvas, Button, CENTER, BOTTOM, BOTH
import tkinter.font as font
import keras.utils
from skimage import color
import numpy
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.optimizers import Adam

directory = os.getcwd()

font_dict = {'fontsize': 6,
             'fontweight': 'bold'}

model_file_type = [('Hierarchical Data Format, H5', '.h5')]
pictures_file_type = [("Картинки", ".jpg .jpeg .png .jpe .bmp"), ('JPG', '.jpg'), ('JPEG', '.jpeg'),
                      ('PNG', '.png'), ('JPE', '.jpe'), ('BMP', '.bmp')]
MODEL_PATH = directory + "\\model.h5"

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


def clear_plot():
    global output
    if output:
        for child in canvas.winfo_children():
            child.destroy()

    output = None


def select():
    global draw_button
    answer = filedialog.askopenfilename(parent=window,
                                        initialdir=directory,
                                        title="Выберите картинку для раскрашивания:",
                                        filetypes=pictures_file_type)
    if not answer:
        return
    draw_button = Button(master=window, command=lambda: colorize(answer), text="Раскрасить")
    draw_button['font'] = button_font
    draw_button.pack(pady=(0, 80))  # Задайте нужные отступы сверху и снизу


def reset_all():
    global back_button, canvas
    back_button.destroy()
    canvas.destroy()
    def_button()


def def_button():
    global select_button, button_font
    select_button = Button(master=window, command=select, text="Выбрать изображение")
    select_button['font'] = button_font
    select_button.pack(pady=(100, 20))


def colorize(path):
    global output, fig, select_button, draw_button, canvas, back_button
    back_button = Button(master=window, command=reset_all, text="Другое изображение")
    back_button['font'] = button_font
    back_button.pack(pady=(0, 0))
    canvas = Canvas(window, width=800, height=500, bg='white')
    canvas.pack()
    clear_plot()
    draw_button.destroy()
    select_button.destroy()
    image = Image.open(path)
    orig_x = image.width
    orig_y = image.height
    image = image.resize((image_width, image_height)).convert("RGB")
    lab = color.rgb2lab(image)
    img_arr = [normalize_l(lab)[:, :, 0]]
    img_arr = numpy.array(img_arr)

    img_arr = img_arr.reshape(img_arr.shape + (1,))
    test_img_unnorm = unnormalize_l(img_arr)
    grayscale = numpy.zeros((image_width, image_height, 3))
    grayscale[:, :, 0] = test_img_unnorm[0][:, :, 0]
    gray_img = Image.fromarray(numpy.uint8(color.lab2rgb(grayscale) * 255))

    output1 = model.predict(img_arr)
    output1 = unnormalize_ab(output1)
    result = numpy.zeros((image_width, image_height, 3))
    result[:, :, 0] = test_img_unnorm[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    color_img = Image.fromarray(numpy.uint8(color.lab2rgb(result) * 255))
    fig = Figure(figsize=(3, 3),
                 dpi=300)

    plot1 = fig.add_subplot(1, 2, 1)

    plot1.imshow(gray_img.resize((orig_x, orig_y)))
    plot1.set_title('Оригинал', font_dict)
    plot1.axis('off')

    plot2 = fig.add_subplot(1, 2, 2)
    plot2.imshow(color_img.resize((orig_x, orig_y)))
    plot2.set_title('Результат', font_dict)
    plot2.axis('off')

    output = FigureCanvasTkAgg(fig, master=canvas)
    output.draw()
    output.get_tk_widget().pack()


window = Tk()
window.title("Раскрашивание чёрно-белых изображений с помощью нейронной сети")
window.resizable(False, False)
window.geometry('800x500')
window.minsize(800, 500)
window['bg'] = "white"

сanvas = None
output = None
fig = None

back_button = None
draw_button = None
select_button = None
button_font = font.Font(family='Comic Sans', size=30, weight="bold")
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
