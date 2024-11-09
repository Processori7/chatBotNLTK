import numpy as np
import json
import random
import nltk
import pymorphy3
import keras
from keras import layers
import os

# Установите уровень логирования TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Скрыть информационные сообщения
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Скрыть информационные сообщения

# Глобальные переменные для хранения данных
morph = pymorphy3.MorphAnalyzer()  # Инициализация морфологического анализатора
words = []  # Список уникальных слов
labels = []  # Список меток (тегов) для намерений
training = []  # Данные для обучения
output = []  # Выходные данные для обучения

def load_intents(file_path):
    # Загрузка намерений из JSON файла
    with open(file_path, encoding='utf-8') as intents:
        return json.load(intents)

def prepare_data(data):
    global words, labels, training, output
    x_docs = []  # Список для хранения токенизированных документов
    y_docs = []  # Список для хранения меток

    # Проход по всем намерениям и их паттернам
    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Токенизация паттерна
            wrds = nltk.word_tokenize(pattern, language='russian')
            words.extend(wrds)  # Добавление токенов в общий список слов
            x_docs.append(wrds)  # Добавление токенов в список документов
            y_docs.append(intent['tag'])  # Добавление метки в список меток

            # Добавление метки в список меток, если её там еще нет
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # Приведение всех слов к нормальной форме и удаление лишних символов
    words = [morph.parse(w)[0].normal_form for w in words if w not in "?"]
    words = sorted(list(set(words)))  # Удаление дубликатов и сортировка
    labels = sorted(labels)  # Сортировка меток

    out_empty = [0 for _ in range(len(labels))]  # Пустой выходной вектор

    # Создание обучающих данных
    for x, doc in enumerate(x_docs):
        bag = []  # Вектор для хранения наличия слов
        wrds = [morph.parse(w)[0].normal_form for w in doc]  # Нормализация слов
        for w in words:
            bag.append(1 if w in wrds else 0)  # Заполнение вектора наличия слов

        output_row = out_empty[:]  # Копирование пустого вектора
        output_row[labels.index(y_docs[x])] = 1  # Установка метки в выходном векторе

        training.append(bag)  # Добавление вектора наличия слов в обучающие данные
        output.append(output_row)  # Добавление выходного вектора в выходные данные

    training = np.array(training)  # Преобразование в массив NumPy
    output = np.array(output)  # Преобразование в массив NumPy

def create_model(input_shape, output_shape):
    # Создание модели нейронной сети
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))  # Входной слой
    model.add(layers.Dense(8, activation='relu'))  # Скрытый слой с 8 нейронами
    model.add(layers.Dense(8, activation='relu'))  # Второй скрытый слой
    model.add(layers.Dense(output_shape, activation='softmax'))  # Выходной слой

    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, training, output, model_file):
    # Обучение модели
    model.fit(training, output, epochs=100, batch_size=8, verbose=1)
    model.save(model_file)  # Сохранение модели в файл
    print("Модель обучена и сохранена в файл.")

def load_model(model_file):
    # Загрузка модели из файла
    return keras.models.load_model(model_file)

def bag_of_words(s, words):
    # Преобразование входной строки в вектор наличия слов
    bag = [0 for _ in range(len(words))]  # Инициализация вектора
    s_words = nltk.word_tokenize(s, language='russian')  # Токенизация входной строки
    s_words = [morph.parse(word)[0].normal_form for word in s_words]  # Нормализация токенов

    # Заполнение вектора наличия слов
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:  # Если слово присутствует в списке слов
                bag[i] = 1  # Устанавливаем 1 в соответствующем индексе

    return np.array(bag)  # Возвращаем вектор в виде массива NumPy


def chat(model, data):
    # Функция для общения с пользователем
    print("Бот готов к общению!! (Введите 'quit' для выхода)")
    while True:
        inp = input("\nВы: ")  # Получение ввода от пользователя
        if inp.lower() == 'quit':  # Проверка на команду выхода
            break

        # Прогнозирование метки на основе входного сообщения
        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)  # Получение индекса с максимальным значением
        tag = labels[results_index]  # Получение метки по индексу

        # Поиск ответа по метке
        for tg in data['intents']:
            if tg['tag'] == tag:  # Если метка совпадает
                responses = tg['responses']  # Получение возможных ответов
                print("Бот: " + random.choice(responses))  # Выбор случайного ответа и вывод


def main():
    # Основная функция
    model_file = 'model.h5'  # Имя файла для сохранения модели
    if os.path.exists('data.json'):  # Проверка наличия файла с данными
        data = load_intents('data.json')  # Загрузка данных
        prepare_data(data)  # Подготовка данных для обучения
        if os.path.exists(model_file):  # Проверка наличия сохраненной модели
            model = load_model(model_file)  # Загрузка модели из файла
            print("Модель загружена из файла.")
        else:
            # Создание и обучение модели, если файл не найден
            model = create_model(len(training[0]), len(output[0]))
            train_model(model, training, output, model_file)
        chat(model, data)  # Запуск функции общения
    else:
        print("Ошибка! Не обнаружен файл data.json")  # Сообщение об ошибке


# Запуск основной функции, если скрипт выполняется напрямую
if __name__ == '__main__':
    main()