# Данная программа автоматически обрабатывает картинки в выбранной папке для дальнейшего использования
# в обучении или тестировании нейронной сети колоризации изображений
# На вход подается аргумент - путь к папке, в которой хранятся картинки
# В этой же папке создастся подпапка с результатом, result
# Всем картинкам даст порядковый номер от 1 до N
# Картинки обязательно фильтруются и изначально чёрно-белые изображения не допускаются
# Картинки перезаписываются в палитре RGB
# Импорт библиотек, sys, os, Image (библиотека Pillow)
import sys
import os
from PIL import Image

PATH = sys.argv[1]  # Получаем из аргумента путь к папке, в которой хранится выборка для обучения
if not os.path.isdir(PATH):  # Если введенного в аргументы пути не существует
    print("Введите существующий путь")  # Выводим информацию, что требуется ввести существующий путь
    exit()  # Завершаем программу
RESULT_PATH = PATH + "\\result"  # Путь к подпапке с результатом

if not os.path.isdir(RESULT_PATH):  # Создаем папку, если ее не существует
    os.mkdir(RESULT_PATH)
i = 1  # Счетчик
for file in os.listdir(PATH):  # Цикл перебора файлов в папке
    image = Image.open(PATH + "\\" + file)  # Создаем объект класса Image, аргумент в конструкторе - путь к файлу,
    # из которого создается изображение
    image = image.resize((480, 360), Image.LANCZOS)  # Изменяем размер изображения до 480 на 360
    image_colorized = Image.new("RGB", image.size)  # Создаем новое цветное изображение с палитрой RGB того же размера
    is_black_and_white = True  # Индикатор того, что изображение полностью чёрно-белое
    try:  # Проверка на ошибки при работе с пикселями изображения. В основном, это то,
        #  что у изображения изначально не 3 цвета
        for x in range(image.width):  # Перебор пикселей по оси X
            for y in range(image.height):  # Перебор пикселей по оси Y
                pixel = image.getpixel(xy=(x, y))  # Получаем пиксель изображения
                if is_black_and_white:  # Если прошлый пиксель
                    is_black_and_white = pixel[0] == pixel[1] == pixel[2]  # Все параметры (r, g, b) у изображения, если
                    # оно чёрно-белое, должны быть равны
                image_colorized.putpixel(xy=(x, y), value=pixel[:3])  # Устанавливаем значение
                # пикселю цветной картинке
        if is_black_and_white:  # Если изображение после перебора всех его пикселей оказалось полностью
            #  чёрно-белым
            print(f'Изображение {file} является чёрно-белым и не подходит для преобразований.')
            continue  # Продолжаем цикл
        image_colorized.save(f'{RESULT_PATH}\\output\\{i}.png')  # Сохраняем измененное цветное изображение в папке с
        # порядковым номером
        print(f'Изображение {file} обработано.')
        i += 1  # Прибавляем к счетчику единицу
    except:
        print(f'Изображение {file} пропущено по ошибке.')  # Выводим информацию, что изображение
        # не обработалось по ошибке
print(f'Преобразование успешно выполнено.')  # Завершение операции
