# Цель работы
Имеется многокомпонентное масло. Известен состав — SMILES молекул входящих в смесь, некоторые свойства компонент по отдельности и масла как целого (свойства смеси). Задача — предсказать определённое свойство масла как целого\
_Задание хакатона Нефтекод_

# Презентация решения
[Google Docs](https://docs.google.com/presentation/d/1PwnA6IwL_MiawJutLk07gsVbz0FZVuLPRJe2TEz3Hqg/edit?usp=sharing)

# Состав репозитория
* `Learning_Beautiful.ipynb` — файл с обучением
* `main.py` — файл для предсказания oil_prperty по датасету
* `descriptors_fun.py` — файл с функциями для вычисления дескрипторов по smiles
* `train_data_fix_concat.csv` и `test_data_fix_concat.csv` — датасеты предоставленные организаторами
* `oil_info_test.csv` и `oil_info_train.csv` — предобработанные датасеты. содеражт теже данные по маслам, что и `train_data_fix_concat.csv` и `test_data_fix_concat.csv`, но в более удобном формате: информация о масле содержится в одной строке, т.е. сгруппирована по `blend_id`
* `*.pkl` — предобученные модели

# Использование main.py

## Установка необходимых библиотек
```
pip install argparse
pip install rdkit
pip install scikit-learn
pip install pandas
pip install numpy
```

## Предсказание
```
python3 main.py -i <input_dataset_path> -o <output_path>
```

Чтобы отобразить прогресс нужно дописать флаг -v

Входной датасет должен иметь такую же структуру как и test_data_fix_concat.csv

## Пример
```
python3 main.py -i test_data_fix_concat.csv -o result.csv -v
```
