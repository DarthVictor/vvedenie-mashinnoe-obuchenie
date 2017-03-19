# -*- coding: utf-8 -*-

import pandas
import numpy
import re
import collections

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
print('data >>>>>> ')
#print(data[:2])

# 1. Какое количество мужчин и женщин ехало на корабле? 
# В качестве ответа приведите два числа через пробел.
print('1. ', 
    numpy.sum(data['Sex'] == 'male'), 
    numpy.sum(data['Sex'] == 'female')
)
# 577 314


# 2. Какой части пассажиров удалось выжить? Посчитайте 
# долю выживших пассажиров. Ответ приведите в процентах 
# (число в интервале от 0 до 100, знак процента не нужен), 
# округлив до двух знаков
print('2. ', 
    round(100.0*numpy.sum(data['Survived'] == 1)/len(data), 2)
)
# 38


# 3. Какую долю пассажиры первого класса составляли среди 
# всех пассажиров? Ответ приведите в процентах (число в 
# интервале от 0 до 100, знак процента не нужен), 
# округлив до двух знаков.
print('3. ', 
    round(100.0*numpy.sum(data['Pclass'] == 1)/len(data), 2)
)
# 24


# 4. Какого возраста были пассажиры? Посчитайте среднее и 
# медиану возраста пассажиров. В качестве ответа приведите
# два числа через пробел.
print('4. ', 
    round(numpy.mean(data['Age'][data['Age'] > 0]), 2),
    round(numpy.median(data['Age'][data['Age'] > 0]), 2)
)
# 29.70 28.00

# 5. Коррелируют ли число братьев/сестер/супругов с числом 
# родителей/детей? Посчитайте корреляцию Пирсона между 
# признаками SibSp и Parch.
print('5. ', 
    round(numpy.corrcoef(data['SibSp'], data['Parch'])[0, 1], 2)
)
# 0.41

# 6. Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя
# (First Name). Это задание — типичный пример того, с чем сталкивается
# специалист по анализу данных. Данные очень разнородные и шумные, 
# но из них требуется извлечь необходимую информацию. Попробуйте вручную 
# разобрать несколько значений столбца Name и выработать правило для 
# извлечения имен, а также разделения их на женские и мужские.

female_names = data[data['Sex']=='female']['Name']
female_first_names = []
for female_name in female_names:
    name_by_brakets = female_name.split('(')
    if len(name_by_brakets) > 1:
        female_first_names.append(re.findall(r"[\w]+", name_by_brakets[1])[0])
    else:
        name_by_miss = re.split(r"Miss|Mrs",  name_by_brakets[0])
        if len(name_by_miss) > 1:
            female_first_names.append(re.findall(r"[\w]+", name_by_miss[1])[0])
        else:
            female_first_names.append(re.findall(r"[\w]+", name_by_miss[0])[0])
# female_first_names = list(set(female_first_names))
# print(female_first_names)
print(collections.Counter(female_first_names).most_common()[:5])
# Mary        
        
        
        
        
        
        
        
        
        
        
        
        
