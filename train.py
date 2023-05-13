"""
Данная программа представляет собой скрипт создания и обучения нейронной сети архитектуры ColorNet
В аргументы запуска скрипта подается путь к обучающей выборке
"""
import os  # Импорт модуля os для работы с операционной системой
import sys  # Импорт модуля sys для работы с системными параметрами и функциями
import random  # Импорт модуля random для работы с генерацией случайных чисел
import time  # Импорт модуля time для работы с функциями времени

from keras.optimizers import Adam  # Импорт класса Adam из модуля optimizers из пакета keras для оптимизации модели
from keras.layers import Conv2D, UpSampling2D, Input, BatchNormalizationV2, Conv2DTranspose  # Импорт определенных классов из модуля layers из пакета keras для создания слоев модели
from keras.models import Model  # Импорт класса Model из модуля models из пакета keras для создания модели
from keras.callbacks import BackupAndRestore, CSVLogger  # Импорт определенных классов из модуля callbacks из пакета keras для обратных вызовов во время обучения модели
from skimage import color  # Импорт модуля color из пакета skimage (Scikit-Image) для работы с цветовыми пространствами
import numpy  # Импорт модуля numpy для работы с массивами
from PIL import Image  # Импорт класса Image из модуля PIL (Python Imaging Library) для работы с изображениями

directory = os.getcwd()  # Получаем текущую рабочую директорию


# Функция создания модели нейронной сети
def create_model():
    inputs = Input((256, 256, 1)) # Входной слой размерности 256x256 и 1 канал
    # Сверточный слой с 64 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(inputs)
    # Сверточный слой с 64 фильтрами, ядром 3x3, шагом 2, сохранением размера, функцией активации ReLU и смещением
    conv1 = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation='relu', use_bias=True)(conv1)
    # Нормализация пакетов
    conv1 = BatchNormalizationV2()(conv1)
    # Сверточный слой с 128 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv1)
    # Сверточный слой с 128 фильтрами, ядром 3x3, шагом 2, сохранением размера, функцией активации ReLU и смещением
    conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation='relu', use_bias=True)(conv2)
    # Нормализация пакетов
    conv2 = BatchNormalizationV2()(conv2)
    # Сверточный слой с 256 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv3 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv2)
    # Сверточный слой с 256 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv3 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv3)
    # Сверточный слой с 256 фильтрами, ядром 3x3, шагом 2, сохранением размера, функцией активации ReLU и смещением
    conv3 = Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation='relu', use_bias=True)(conv3)
    # Нормализация пакетов
    conv3 = BatchNormalizationV2()(conv3)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv4 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv3)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv4 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv4)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv4 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv4)
    # Нормализация пакетов
    conv4 = BatchNormalizationV2()(conv4)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv5 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv4)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv5 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv5)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv5 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv5)
    # Нормализация пакетов
    conv5 = BatchNormalizationV2()(conv5)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv6 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv5)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv6 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv6)
    # Сверточный слой с 512 фильтрами, ядром 3x3, разрежением нейрона 2, шагом 1,
    # сохранением размера и функцией активации ReLU
    conv6 = Conv2D(filters=512, kernel_size=3, dilation_rate=2, strides=1, padding="same", activation='relu',
                   use_bias=True)(conv6)
    # Нормализация пакетов
    conv6 = BatchNormalizationV2()(conv6)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv7 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv6)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv7 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv7)
    # Сверточный слой с 512 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv7 = Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv7)
    # Нормализация пакетов
    conv7 = BatchNormalizationV2()(conv7)
    # Транспонированный слой свертки (развертка)
    # с 256 фильтрами, ядром 4x4, шагом 2, сохранением размера, функцией активации ReLU и смещением
    conv8 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same", activation='relu', use_bias=True)(
        conv7)
    # Сверточный слой с 256 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv8 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv8)
    # Сверточный слой с 256 фильтрами, ядром 3x3, шагом 1, сохранением размера, функцией активации ReLU и смещением
    conv8 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu', use_bias=True)(conv8)
    # Сверточный слой с 256 фильтрами, ядром 1x1, шагом 1, без дополнения нулями (потеря размера),
    # функцией активации SoftMax и смещением
    conv8 = Conv2D(filters=313, kernel_size=1, strides=1, padding="valid", activation='softmax', use_bias=True)(conv8)
    # Сверточный слой с 2 фильтрами, ядром 1x1, шагом 1, разрежением нейрона 1, без дополнения нулями (потеря размера),
    # функцией активации SoftMax и без смещения
    conv9 = Conv2D(filters=2, kernel_size=1, strides=1, dilation_rate=1, padding="valid", use_bias=False)(conv8)
    # Слой увеличения размерности изображения с использование метода бикубической интерполяции
    # слой увеличивает размеры входа в 2 раза в каждом измерении
    output = UpSampling2D(size=4, interpolation="bicubic")(conv9)
    # Создание экземпляра модели используя записанные ранее входы и выходы. Т.е. на вход модель получает
    # inputs, а на выходе выдает output
    model = Model(inputs=inputs, outputs=output)
    # Определяется объект оптимизатора Adam с определенным шагом обучения 0.0001
    adamOpti = Adam(learning_rate=0.0001)
    # Компиляция модели нейронной сети. Ее оптимизатор - ранее созданный Адам, функция потери mse
    # и обязательный подсчет точности.
    model.compile(optimizer=adamOpti, loss="mse", metrics=["accuracy"])
    # Эта фукнция выводит в консоль сводку архитектуры модели,
    # включая число параметров и форму выходных данных каждого слоя.
    model.summary()
    # Функция возвращает модель, созданную в этой функции, чтобы ее можно было использовать в других частях программы
    return model


batch = 64  # Количество изображений в одном пакете (батче). Проще,
# сколько изображений за раз получает модель для обучения
epoch = 250  # Количество эпох обучения нейронной сети
step_ep = None # В эту переменную будет записано количество шагов, которые нужно сделать генератору
# перед уходом на следующую эпоху обучения
train_list = None  # Переменная, куда запишется список файлов для обучения
PATH = sys.argv[1]  # Получаем из аргумента путь к папке, в которой хранится выборка для обучения
if not os.path.isdir(PATH):  # Если введенного в аргументы пути не существует
    print("Введите существующий путь")  # Выводим информацию, что требуется ввести существующий путь
    exit()  # Завершаем программу


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


# Определяем функцию-генератор, которая будет возвращать набор данных пакетами указанного размера
def generator():
    i = 0
    while True:
        x = []  # Создаем пустой список входных значений
        y = []  # Создаем пустой список выходных значений
        for b in range(batch):  # Перебираем изображения в папке с выборкой, пока не наберем нужное количество
            if i == len(train_list):  # Если прошли все изображения в списке, перемешиваем их и начинаем заново
                i = 0
                random.shuffle(train_list)
            img = train_list[i]  # Выбираем изображение по индексу i
            i += 1  # Увеличиваем счетчик изображений
            rgb_image = Image.open(os.path.join(PATH, img)).resize(
                (256, 256))  # Открываем изображение и изменяем его размер
            lab = color.rgb2lab(rgb_image)  # Преобразуем RGB-изображение в формат Lab*
            norm_l = normalize_l(lab)  # Нормализуем параметр L
            norm_ab = normalize_ab(lab)  # Нормализуем параметры a и b
            x.append(norm_l[:, :, 0])  # Добавляем нормализованный параметр L в список входных значений
            y.append(norm_ab[:, :, 1:])  # Добавляем нормализованные параметры a и b в список выходных значений
        x = numpy.array(x)  # Преобразуем список входных значений в numpy array
        y = numpy.array(y)  # Преобразуем список выходных значений в numpy array
        yield x, y  # Возвращаем пакет нормализованных изображений


model = create_model()  # Создаем модель нейронной сети
train_list = list(os.listdir(path=PATH))  # Указываем путь к обучающей выборке
random.shuffle(train_list)  # Перемешиваем список с именами изображений
step_ep = len(train_list) // batch  # Рассчитываем количество шагов на каждой эпохе

cur_time = time.time()  # Записываем текущее время
print("Всего найдено", len(train_list), "изображений")  # Создаем объект для резервного копирования модели
callback = BackupAndRestore(backup_dir=directory + "\\backup\\", delete_checkpoint=False)  # Создаем объект
# для резервного копирования модели
csv_logger = CSVLogger(directory + "\\model_history_log.csv", append=True) # Создаем объект
# для записи журнала обучения в CSV-файл
print("Начинаем обучение...")  # Выводим сообщение о начале обучения
# Обучаем нейронную сеть. За раз обучаем 64 фотографиям по 250 эпох
history = model.fit(generator(), epochs=epoch, batch_size=batch, verbose=1, steps_per_epoch=step_ep,
                    callbacks=[callback, csv_logger])
print("Затраченное время", round(time.time() - cur_time, 2), "сек.")  # Выводим затраченное время на обучение
model.save("model.h5", save_format="h5")  # Сохраняем модель в файл
