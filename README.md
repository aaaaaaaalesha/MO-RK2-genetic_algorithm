# Исследование генетических алгоритмов в задачах поиска экстремумов.

<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

#### График поверхности:

![graphic](https://user-images.githubusercontent.com/55093100/98445106-e9ce0900-2126-11eb-8582-f818c1d0ff89.png)

#### Визуализация генетического алгоритма:

![visualisation](https://user-images.githubusercontent.com/55093100/98445095-da4ec000-2126-11eb-94b0-5cf822a74144.gif)

### Начальное и первые 10 поколений:

```
Поколение: 0
+-----------+-----------+-----------+
|     X     |     Y     |    FIT    |
+-----------+-----------+-----------+
|  1.723671 |  -1.07902 |  0.192458 |
| -1.894173 | -0.240393 | -0.204097 |
|  1.723671 | -0.638812 |  0.225693 |
|  0.348256 | -0.240393 |  0.28943  |
+-----------+-----------+-----------+
Максимальный результат: 0.28943
Средний результат: 0.125871

Поколение: 1
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.348256 |  -1.07902 | 0.14931  |
| 1.723671 | -0.240393 | 0.245316 |
| 0.348256 | -0.638812 | 0.223138 |
| 1.723671 | -0.240393 | 0.245316 |
+----------+-----------+----------+
Максимальный результат: 0.245316
Средний результат: 0.21577

Поколение: 2
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 1.723671 | -0.638812 | 0.225693 |
| 0.348256 | -0.240393 | 0.28943  |
| 1.723671 | -0.240393 | 0.245316 |
| 1.723671 | -0.240393 | 0.245316 |
+----------+-----------+----------+
Максимальный результат: 0.28943
Средний результат: 0.251439

Поколение: 3
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.348256 | -0.240393 | 0.28943  |
| 1.723671 | -0.240393 | 0.245316 |
| 0.348256 | -0.240393 | 0.28943  |
| 1.723671 | -0.240393 | 0.245316 |
+----------+-----------+----------+
Максимальный результат: 0.28943
Средний результат: 0.267373

Поколение: 4
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 1.662667 | -0.147563 | 0.263001 |
| 0.406113 | -0.159661 | 0.331851 |
| 1.662667 | -0.296956 | 0.258467 |
| 0.447112 | -0.159661 | 0.352834 |
+----------+-----------+----------+
Максимальный результат: 0.352834
Средний результат: 0.301538

Поколение: 5
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 | -0.190425 | 0.358639 |
| 1.756958 |  -0.23524 | 0.237244 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.381655 |  -0.23524 | 0.310123 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.316718

Поколение: 6
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 |  -0.23524 | 0.353264 |
| 0.381655 | -0.168883 | 0.317205 |
| 0.466373 | -0.190425 | 0.358639 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.347494

Поколение: 7
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 |  -0.23524 | 0.353264 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.190425 | 0.358639 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.35841

Поколение: 8
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 | -0.190425 | 0.358639 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.36031

Поколение: 9
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.360867

Поколение: 10
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.360867
```

### Поколения N = 10...100 с шагом 10:

```
Поколение: 10
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
| 0.466373 | -0.168883 | 0.360867 |
+----------+-----------+----------+
Максимальный результат: 0.360867
Средний результат: 0.360867

Поколение: 20
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.554139 | -0.185469 | 0.392265 |
| 0.554139 | -0.159542 | 0.394898 |
| 0.554139 | -0.159542 | 0.394898 |
| 0.554139 | -0.159542 | 0.394898 |
+----------+-----------+----------+
Максимальный результат: 0.394898
Средний результат: 0.39424

Поколение: 30
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.588973 | -0.066582 | 0.411084 |
| 0.588286 | -0.066582 | 0.410907 |
| 0.588973 | -0.066582 | 0.411084 |
| 0.588973 | -0.066582 | 0.411084 |
+----------+-----------+----------+
Максимальный результат: 0.411084
Средний результат: 0.41104

Поколение: 40
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.627848 | -0.185639 | 0.411159 |
| 0.502763 | -0.087963 | 0.382265 |
| 0.627848 | -0.082042 | 0.419298 |
| 0.563936 | -0.087963 | 0.403177 |
+----------+-----------+----------+
Максимальный результат: 0.419298
Средний результат: 0.403975

Поколение: 50
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.764859 |  -0.16668 | 0.42934  |
| 0.706055 | -0.016062 | 0.432913 |
| 0.764859 | -0.124542 | 0.432631 |
| 0.706055 | -0.016062 | 0.432913 |
+----------+-----------+----------+
Максимальный результат: 0.432913
Средний результат: 0.431949

Поколение: 60
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.821945 |  0.05469  | 0.436362 |
| 0.745884 | -0.113321 | 0.432468 |
| 0.821945 |  0.031595 | 0.436881 |
| 0.673733 | -0.113321 | 0.425365 |
+----------+-----------+----------+
Максимальный результат: 0.436881
Средний результат: 0.432769

Поколение: 70
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.797664 | -0.109481 | 0.434233 |
| 0.850419 |  0.053489 | 0.435414 |
| 0.797664 |  0.053489 | 0.436651 |
| 0.785819 |  0.053489 | 0.436569 |
+----------+-----------+----------+
Максимальный результат: 0.436651
Средний результат: 0.435717

Поколение: 80
+----------+----------+----------+
|    X     |    Y     |   FIT    |
+----------+----------+----------+
| 0.797664 | 0.053489 | 0.436651 |
| 0.797664 | 0.053489 | 0.436651 |
| 0.797664 | 0.053489 | 0.436651 |
| 0.797664 | 0.053489 | 0.436651 |
+----------+----------+----------+
Максимальный результат: 0.436651
Средний результат: 0.436651

Поколение: 90
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.766477 |  0.154315 | 0.430461 |
| 0.917409 | -0.037012 | 0.430833 |
| 0.766477 | -0.037012 | 0.436541 |
| 0.886586 | -0.037012 | 0.433544 |
+----------+-----------+----------+
Максимальный результат: 0.436541
Средний результат: 0.432845

Поколение: 100
+----------+-----------+----------+
|    X     |     Y     |   FIT    |
+----------+-----------+----------+
| 0.766477 | -0.037012 | 0.436541 |
| 0.766477 | -0.037012 | 0.436541 |
| 0.766477 | -0.037012 | 0.436541 |
| 0.766477 | -0.037012 | 0.436541 |
+----------+-----------+----------+
Максимальный результат: 0.436541
Средний результат: 0.436541
```
