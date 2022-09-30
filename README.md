# Сегментация капилляров глаза человека по снимкам с офтальмологической щелевой лампы

## Обучение

Эксперименты проводились на основе следующих решений:
* [RVGAN](https://github.com/SharifAmit/RVGAN "GitHub")
* [SGL-Retinal-Vessel-Segmentation](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation "GitHub")
* [SA-UNet](https://github.com/clguo/SA-UNet "GitHub")

Также был разработан собственный пайплайн для обучения моделей из [SMP](https://github.com/qubvel/segmentation_models.pytorch "GitHub").\
Код расположен в директории: `./training/SegmentationTrainingTiling/`

Наилучшие результаты на тестовой выборке (загрузка на сайт) продемонстрировала модернизированная реализация из репозитория:
* [MedISeg](https://github.com/hust-linyi/MedISeg "GitHub")

В частности были внесены следующие изменения:
* Загрузку данных переделал для генерации кропов из полного изображения
* Код инференс модели переделал под проход изображения окном с паресеченими
* Добавил постобработку результатов подели, а в частности фильтрация по контурам

Изменённый код расположен по пути: `./training/MedISeg/`

### Запуск обучения

#### Конвертация данных
Для загрузки данных пользовался следующим форматом представления данных:
```shell
├── images
│   ├── IMG_1.png
│   ├── IMG_2.png
│   ├── ...
│   └── IMG_N.png
└── masks
    ├── IMG_1.png
    ├── IMG_2.png
    ├── ...
    └── IMG_N.png
```
Чтобы сконвертировать входной набор изображений с разметкой в `.geojson` формате, можно воспользоваться следующим скриптом:
`training/scripts/convert_dataset_to_masks.py`. Описание к нему:
```shell
usage: convert_dataset_to_masks.py [-h] [-i INPUT] [-o OUTPUT]

Convert dataset with geojson to image+mask format

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input folder
  -o OUTPUT, --output OUTPUT
                        Path to output folder with images/ and masks/ folders
```
#### Фильтрация данных
Данные были разделены на тренировочную и валидационную выборку с соотношением `0.8` и `0.2` соответственно.\
Затем вручную выбрал изображения, содержащие наибольшее количество размеченных сосудов.
Профильтрованный набора данных можно скачать по [следующей ссылке](https://disk.yandex.ru/d/cac9w9lM2z3lKA "Yandex Disk")


## Веб интерфейс и микросервис
Обученную модель интегрировал в микросервис, который обрабатывает http запросы.\
Такая реализация позволяет легко интегрировать данное решение в различные реализаци интерфейсов. Буть то **веб** интерфейс или десктопное приложение, а также и в **IoT** устройства.

### Инструкция запуска через Docker
#### Версия без GPU (CPU only)
Для удобства использования решения, микросервис и веб интерфейс можно запустить через Docker.

1. Для сборки Docker image воспользуйтесь следующей командой:
```shell
docker build -t vessel_segmentation .
```

2. Для запуска Docker контейнера воспользуйтесь следующей командой:
```shell
docker run --restart=always \
 -p 9009:9009 -p 8051:8051 \
 --name "EyeVesselSegmentation"\
 -d vessel_segmentation:latest
```

На 9009 порту будет запущен микросервис, а на 8051 веб интерфейс.

Для тестирования микросервиса напрямую, можно воспользоваться следующей командой:
```shell
curl -F image=@Image.png http://localhost:9009/predict -o res_mask.webp
```

***Note:*** Для более эффективного запуска модели на CPU можно сконвертировать её в ONNX

#### Версия с GPU (CUDA)
Для этого необходимо поменять первую строчку в `Dockerfile` чтобы использовать базовый образ от Nvidia. В нём же есть инструкция.\
Команда для сборки останется прежней, а для запуска поменяется на следующую:
```shell
docker run --restart=always --runtime=nvidia \
 -p 9009:9009 -p 8051:8051 \
 --name "EyeVesselSegmentation"\
 -d vessel_segmentation:latest
```