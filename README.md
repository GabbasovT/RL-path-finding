# RL-path-finding

## Описание проекта

RL-path-finding - это проект, реализующий алгоритм поиска пути с использованием обучения с подкреплением (Reinforcement Learning, RL). В основе проекта лежит алгоритм TD3 (Twin Delayed Deep Deterministic Policy Gradient), который является современным методом для задач непрерывного управления.

## Алгоритм TD3

TD3 (Twin Delayed Deep Deterministic Policy Gradient) - это алгоритм обучения с подкреплением, который улучшает оригинальный DDPG (Deep Deterministic Policy Gradient). Основные особенности TD3:

1. **Двойные Q-сети (Twin)**: Используются две отдельные Q-сети для уменьшения переоценки значений.
2. **Задержка обновления политики (Delayed)**: Политика обновляется реже, чем Q-функции, для стабильности.
3. **Добавление шума в целевое действие (Target Policy Smoothing)**: Регуляризация, предотвращающая переобучение.

TD3 особенно эффективен в задачах с непрерывным пространством действий, таких как управление роботами или, как в нашем случае, поиск пути.

## Установка зависимостей

Перед использованием проекта необходимо установить следующие зависимости:

### 1. Установка LibTorch

LibTorch - это C++ версия PyTorch, необходимая для работы нейронных сетей.

1. Перейдите на страницу загрузки PyTorch: https://pytorch.org/get-started/locally/#libtorch
2. Выберите версию LibTorch (рекомендуется последняя стабильная).
3. Для Linux выберите вариант "C++/Java" и загрузите версию для вашей системы.
4. Распакуйте архив в удобное место, например:
   ```bash
   unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cpu.zip -d ~/RL-path-finding/libtorch/
   ```

### 2. Установка SFML 3.0.0

SFML (Simple and Fast Multimedia Library) используется для визуализации.

1. Перейдите на страницу загрузки SFML: https://www.sfml-dev.org/download/sfml/3.0.0/
2. Скачайте версию для вашего дистрибутива Linux.
3. Установите зависимости:
   ```bash
   sudo dnf install SFML-devel
   ```
4. Распакуйте архив в удобное место, например:
   ```bash
   tar -xzf SFML-3.0.0-linux-gcc-64-bit.tar.gz -С ~/RL-path-finding/SFML-3.0.0/
   ```

## Возможности

Программа предоставит интерфейс для:
- Обучения модели TD3
- Тестирования обученной модели
- Визуализации процесса поиска пути

## Структура проекта

```
RL-path-finding/
├── config/                 # Конфигурационный файл
│   └── Config.h            # все гиперпараметры тут
├── include/                # Заголовочные файлы
│   ├── environment/
│   │   ├── Env.hpp
│   │   └── Renderer.hpp
│   └── ml/
│       └── RL.hpp
├── libtorch/           # LibTorch
├── SFML-3.0.0/         # SFML
├── src/                    # Исходный код
│   ├── common/             # Общие компоненты
│   │   ├── Consts.hpp
│   │   ├── Enums.hpp
│   │   └── Types.hpp
│   ├── environment/        # Логика окружения и графики
│   │   ├── CMakeLists.txt
│   │   ├── Env.cpp
│   │   └── Renderer.cpp
│   ├── ml/                 # Реализация RL
│   │   ├── CMakeLists.txt
│   │   └── RL.cpp
│   ├── main.cpp            # Точка входа
│   └── CMakeLists.txt      # Основной CMake файл
└── README.md
```

## Примеры работы

### Обучение агента
![Пример обучения](images/start.png)
![Пример обучения](images/training.png)

### Видео с результатами
[Видео с результатами](images/video.mp4)

## Настройка параметров

Параметры обучения и окружения можно настроить в файле `config/Config.h`:
```cpp
// Параметры обучения
const int EPISODES = 10000;
const int MAX_STEPS = 500;
const int BATCH_SIZE = 512;
const int LOG_INTERVAL = 50;
const float ACTOR_LR = 3e-5;
const float CRITIC_LR = 3e-5;
const float GAMMA = 0.99f;          // Коэффициент дисконтирования
const float TAU = 0.005f;           // Для мягкого обновления целевых сетей
const int TRAIN_START_SIZE = 5000;
const int TRAIN_INTERVAL = 1;
```

## Разработчики

Габбасов Тимур ```GabbasovT```
Войко Артём ```LanavaX```
