"""
Код приложения для работы с обученной моделью нейронной сети
"""
import os  # Импорт модуля os для работы с операционной системой
import tkinter as tk  # Импорт модуля tkinter для создания графического интерфейса
from tkinter import messagebox, filedialog  # Импорт определенных классов из модуля tkinter
import tkinter.font as font  # Импорт класса font из модуля tkinter
import numpy as np  # Импорт модуля numpy для работы с массивами
from PIL import Image  # Импорт класса Image из модуля PIL (Python Imaging Library)
from skimage import color  # Импорт модуля color из пакета skimage (Scikit-Image)
from matplotlib.figure import Figure  # Импорт класса Figure из модуля figure из пакета matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk  # Импорт определенных классов из модуля backend_tkagg и NavigationToolbar2Tk из пакета
# matplotlib
from tensorflow import keras  # Импорт модуля keras из пакета tensorflow
from keras.optimizers import Adam  # Импорт класса Adam из модуля optimizers из пакета keras

MODEL_PATH = "model.h5"  # Путь к файлу модели
IMAGE_WIDTH = 256  # Ширина изображения
IMAGE_HEIGHT = 256  # Высота изображения


def normalize_l(in_l):  # Функция нормализации компонента L (яркости)
    # Она нормализирует L, который измеряется от 0 до 100 до (-0.5;0.5)
    return (in_l - 50) / 100


def unnormalize_l(in_l):  # Функция денормализации компонента L (яркости)
    # Она денормализует L из (-0.5;0.5) в значения от 0 до 100
    return in_l * 100 + 50


def normalize_ab(in_ab):  # Функция нормализации компонентов a и b (-128;128) до
    #  (-1;1) диапазона
    return in_ab / 110


def unnormalize_ab(in_ab):  # Функция денормализации компонентов a и b
    #  из (-1;1) в (-128;128)
    return in_ab * 110


# Функция выбора изображения
def select_image():
    # Диалоговое окно с выбором файла изображения определенных форматов
    answer = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Выберите картинку для раскрашивания:",
        filetypes=[("Картинки", ".jpg .jpeg .png .jpe .bmp"), ('JPG', '.jpg'), ('JPEG', '.jpeg'),
                   ('PNG', '.png'), ('JPE', '.jpe'), ('BMP', '.bmp')]
    )
    if answer:  # Если пользователь выбрал файл
        label_selected = tk.Label(master=window, text="Выбрано: " + answer)  # Создание виджета Label для отображения
        # пути выбранного файла
        label_selected.pack(pady=(0, 80))  # Размещение виджета на главном окне с отступом
        draw_button = tk.Button(master=window, command=lambda: colorize_image(answer), text="Раскрасить")
        # Создание кнопки для вызова функции раскрашивания изображения
        draw_button['font'] = button_font  # Настройка шрифта кнопки
        draw_button.pack(pady=(0, 80))  # Размещение кнопки на главном окне с отступом


# Функция удаления всех виджетов на главном окне
def destroy_all():
    for widget in window.winfo_children():
        widget.destroy()


# Функция удаления всех виджетов и создания кнопки по умолчанию "Выбрать изображение"
def destroy_all_and_create_default():
    destroy_all()
    def_button()


# Функция создания кнопки "Выбрать изображение", которая открывает пользователю выбор изображения для раскраски
def def_button():
    select_button = tk.Button(
        master=window,
        command=select_image,
        text="Выбрать изображение",
        font=button_font
    )
    select_button.pack(pady=(100, 20))


# Функция раскрашивания изображения, параметр - путь к изображению
def colorize_image(path):
    destroy_all()  # Удаление всех виджетов на главном окне
    #  Создание кнопки "Другое изображение", которая позволяет выбрать другое изображение для раскрашивания
    back_button = tk.Button(
        master=window,
        command=destroy_all_and_create_default,
        text="Другое изображение",
        font=button_font
    )
    back_button.pack(pady=(0, 0))

    # Создание полотна Canvas для отображения pyplot
    canvas = tk.Canvas(window, bg='white')
    canvas.pack()

    # Открытие изображения и изменение его размера
    image = Image.open(path)
    orig_x, orig_y = image.size
    # Изменение размера изображения до заданных размеров (IMAGE_WIDTH, IMAGE_HEIGHT) и преобразование в цветовое
    # пространство RGB
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert("RGB")
    # Преобразование изображения в цветовое пространство Lab
    lab = color.rgb2lab(image)
    # Нормализация канала L
    img_arr = [normalize_l(lab)[:, :, 0]]  # Извлечение и нормализация канала L
    img_arr = np.array(img_arr)

    img_arr = img_arr.reshape(img_arr.shape + (1,))  # Добавление размерности для совместимости с моделью
    test_img_unnorm = unnormalize_l(img_arr)  # Восстановление нормализованного канала L
    grayscale = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    grayscale[:, :, 0] = test_img_unnorm[0][:, :, 0]  # Извлечение канала L для получения черно-белого изображения
    gray_img = Image.fromarray(
        np.uint8(color.lab2rgb(grayscale) * 255))  # Преобразование обратно в RGB и создание черно-белого изображения

    output1 = model.predict(img_arr)  # Получение предсказанных значений для каналов ab
    output1 = unnormalize_ab(output1)  # Восстановление нормализованных каналов ab
    result = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    result[:, :, 0] = test_img_unnorm[0][:, :, 0]  # Извлечение канала L для результирующего изображения
    result[:, :, 1:] = output1[0]  # Добавление предсказанных значений для каналов ab в результирующее изображение
    color_img = Image.fromarray(
        np.uint8(color.lab2rgb(result) * 255))  # Преобразование результирующего изображения в RGB

    plot_figure = Figure(figsize=(6, 3), dpi=300)  # Создание фигуры для графиков
    # Создание первого подграфика и отображение изображения
    plot1 = plot_figure.add_subplot(1, 3, 1)
    plot1.imshow(image.resize((orig_x, orig_y)))

    # Установка заголовка и параметров отображения для первого подграфика
    plot1.set_title('Оригинал', fontsize=6, fontweight='bold')
    plot1.axis('off')
    plot1.set_aspect('equal')

    # Создание второго подграфика и отображение черно-белого изображения
    plot2 = plot_figure.add_subplot(1, 3, 2)
    plot2.imshow(gray_img.resize((orig_x, orig_y)))

    # Установка заголовка и параметров отображения для второго подграфика
    plot2.set_title('Черно-белое', fontsize=6, fontweight='bold')
    plot2.axis('off')
    plot2.set_aspect('equal')

    # Создание третьего подграфика и отображение цветного изображения
    plot3 = plot_figure.add_subplot(1, 3, 3)
    plot3.imshow(color_img.resize((orig_x, orig_y)))

    # Установка заголовка и параметров отображения для третьего подграфика
    plot3.set_title('Результат', fontsize=6, fontweight='bold')
    plot3.axis('off')
    plot3.set_aspect('equal')

    # Создание объекта FigureCanvasTkAgg и отрисовка изображения на холсте
    output = FigureCanvasTkAgg(plot_figure, master=canvas)
    output.draw()
    output.get_tk_widget().pack(expand=True)

    # Создание панели инструментов и ее размещение на окне
    toolbar = NavigationToolbar2Tk(output, window)
    toolbar.update()
    toolbar.pack()


window = tk.Tk()  # Создание главного окна приложения
window.title("Раскрашивание чёрно-белых изображений с помощью нейронной сети")  # Установка заголовка окна
window.geometry('1920x1080')  # Установка размеров окна
window.minsize(1080, 720)  # Установка минимальных размеров окна
window['bg'] = "white"  # Установка фона окна

button_font = font.Font(family='Comic Sans', size=25, weight="bold")  # Создание шрифта для кнопок

def_button()  # Вызов функции для создания кнопок в окне

if not os.path.isfile(MODEL_PATH):  # Проверка наличия файла модели нейронной сети
    messagebox.showinfo("Ошибка", "Модель нейронной сети не найдена. Убедитесь, что файл model.h5 лежит в одной "
                                  "директории с файлом запуска приложения.")  # Вывод сообщения об ошибке
    exit()  # Выход из программы
else:
    model = keras.models.load_model(MODEL_PATH)  # Загрузка модели нейронной сети из файла
    adamOpti = Adam(learning_rate=0.0001)  # Создание оптимизатора Adam
    model.compile(optimizer=adamOpti, loss="mse", metrics=["accuracy"])  # Компиляция модели с выбранными параметрами

window.mainloop()  # Запуск главного цикла окна приложения
