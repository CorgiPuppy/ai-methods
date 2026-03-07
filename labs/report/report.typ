#show raw.where(block: true): it => { set par(justify: false); grid(
  columns: (100%, 100%),
  column-gutter: -100%,
  block(width: 100%, inset: 1em, for i in it.text.split("\n") {
    linebreak()
  }),
  block(radius: 1em, fill: luma(246), width: 100%, inset: 1em, it),
)}
#show table.cell.where(y: 0): strong

#set page(
	paper: "a4",
	margin: (x: 0.8cm, y: 1.5cm),
)

#set text(
	font: "Times New Roman",
	size: 11pt,
)

#set par(first-line-indent: (
	amount: 2em,
	all: true,
))

#set page(numbering: none)
#align(center, block[
	#set text(size: 12pt)
	#set heading(outlined: false)
	= Министерство науки и высшего образования Российской Федерации 
	= Российский химико-технологический университет имени Д.И. Менделеева
	#line(length: 100%, stroke: 0.5pt)
	= Факультет цифровых технологий и химического инжиниринга
	= Кафедра информационных компьютерных технологий
])

#v(15%)

#align(center, block[
	#set text(size: 16pt)
	#set heading(outlined: false)
	= ОТЧЕТ
	= ПО ЛАБОРАТОРНОМУ ПРАКТИКУМУ 
	#v(1em)
	#set text(size: 14pt)
	= «Методы искусственного интеллекта»
	#v(2em)
	= Вариант №1
])

#v(10%)

#align(left)[
	#table(
		columns: (8.5cm, 10.8cm),
		stroke: none,
		table.cell(text(size: 13.4pt, "ВЫПОЛНИЛ:"), align: right), 
		table.cell(text(size: 14.4pt, "студент группы КС-46 Золотухин А.А."), align: left),
		
		table.cell(text(size: 14.4pt, "ПРОВЕРИЛ:"), align: right), 
		table.cell(text(size: 14.4pt, "ассистент Макляев И.В."), align: left),
	)
]

#v(12%)

#align(center, block[
	#set text(size: 14.4pt)
	#set heading(outlined: false)
	= Москва
	= 2026
])

#pagebreak()

#align(center, block[
	#outline(		
		title: "СОДЕРЖАНИЕ",
	)
])

#pagebreak()

#set page(numbering: "1")
#align(center, block[
	= 1. Лабораторная работа 1. Решение задачи регрессии
])

#align(center, block[
	== 1.1. Цель работы. Задача. Вариант задания
])
*Цель работы:* приобретение навыков анализа данных, проектирования и обучения искусственных нейронных сетей для решения задач аппроксимации и регрессии с использованием фреймворка Keras.

*Задача:* На основе набора данных «Diamonds» построить модель прогнозирования рыночной стоимости бриллианта в зависимости от его характеристик.

*Вариант задания:* Вариант 1 (Задание на регрессию).

#align(center, block[
	== 1.2. Теоретическая часть
])

#align(center, block[
	== 1.3. Практическая часть
])

#align(center, block[
	=== 1.3.1. Описание данных
])
Первым этапом работы было исследование структуры набора данных «Diamonds».

*Первые строки датасета:*
#block(fill: luma(240), inset: 10pt, radius: 5pt, width: 100%)[
	#set text(size: 8pt, font: "Courier New")
	#raw(read("../lab1/assets/data_head.txt"))
]

*Общая информация о типах данных:*
#block(fill: luma(240), inset: 10pt, radius: 5pt, width: 100%)[
	#set text(size: 8pt, font: "Courier New")
	#raw(read("../lab1/assets/data_info.txt"))
]

#align(center, block[
	=== 1.3.2. Предварительный анализ данных (EDA)
])
Для оценки качества признаков и их распределения были построены следующие графики:

#grid(
	columns: (1fr, 1fr),
	gutter: 10pt,
	figure(image("../lab1/assets/price_dist.png"), caption: [Распределение целевой переменной Price], supplement: [Рис.]),
	figure(image("../lab1/assets/heatmap.png"), caption: [Матрица корреляций], supplement: [Рис.])
)

*Анализ:* На графике распределения видно, что большинство цен сосредоточено в диапазоне до 5000\$.

#figure(
	image("../lab1/assets/price_carat.png", width: 60%),
	caption: [Взаимосвязь веса в каратах и стоимости],
	supplement: [Рис.]
)

В ходе анализа были визуализированы распределения признаков и построена матрица корреляций.

#figure(
	image("../lab1/assets/heatmap.png", width: 80%),
	caption: [Матрица корреляций признаков],
	supplement: [Рис.]
)

Выявлено, что на тепловой карте подтверждается сильная корреляция (0.88+) цены с весом (`carat`) и физическими размерами (`x`, `y`, `z`).

#align(center, block[
	=== 1.3.3. Обучение моделей и результаты
])
Было проведено обучение двух нейронных сетей: однослойного перцептрона и многослойного (MLP) с двумя скрытыми слоями по 64 нейрона. Для обучения использовался оптимизатор Adam. Результаты сравнения моделей на тестовой выборке представлены в таблице ниже.

#align(center)[
	#table(
		columns: (auto, auto, auto, auto),
		inset: 10pt,
		[*Модель*], [*MSE*], [*MAE*], [*R^2*],
		[Однослойная], [3.01e+07], [3810.63], [-0.9284],
		[Многослойная (MLP)], [1.23e+06], [617.67], [0.9211],
	)
]    

*Примечание:* Отрицательный $R^2$ у однослойной модели говорит о том, что она предсказывает хуже, чем просто среднее значение по выборке, что подтверждает нелинейность данных.

#figure(
	image("../lab1/assets/training_comparison.png", width: 100%),
	caption: [Сравнение графиков обучения: Однослойная (слева) и Многослойная (справа)],
	supplement: [Рис.]
)

#align(center, block[
	== 1.4. Выводы по работе
])
1. В ходе работы было проведено исследование данных о бриллиантах. Удаление неинформативных признаков (индексов) и нормализация данных позволили добиться стабильного обучения.
2. Сравнение показало, что многослойный перцептрон на порядок превосходит однослойный по всем метрикам. Коэффициент $R^2 = 0.91$ указывает на высокую точность предсказания MLP.
3. Однослойная структура не справилась с задачей, что подтверждает необходимость использования скрытых слоев для моделирования сложных рыночных зависимостей.
