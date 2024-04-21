import argparse

import pandas as pd
import numpy as np
import joblib #для загрузки обученных моделей

from rdkit import Chem #для работы со smiles
from rdkit.Chem import MACCSkeys #для создания fingerprints молекул
from descriptors_fun import n_mol, n_atom_mean, frac_atom_с, frac_atom_o, mol_w_mean, MolLogP_mean, TPSA_sum, n_bonds_mean, n_bonds_nn_mean, n_col, ind_V_mean, LabuteASA, HallKierAlpha #функции вычисления определённых дескрипторов по smiles


parser = argparse.ArgumentParser(description='Предсказание параметра масла по smiles молекул входящих в состав и свойствах масла')

parser.add_argument('-i', '--input', dest='input_file_path', required=True,
                    help='путь к файлу с датасетом для предсказаний')
parser.add_argument('-o', '--output', dest='output_file_path', required=True,
                    help='путь к файлу для сорхранения результатов')
parser.add_argument('-v', '--verbose', action='store_true',
                        help='показывать ход выполнения')

args = parser.parse_args()
file_with_test_dataset = args.input_file_path # "./test_data_fix_concat.csv"
output_file = args.output_file_path
verbose = args.verbose

### Чтение датасета
if verbose:
    print("Чтение датасета")
    
df0 = pd.read_csv(file_with_test_dataset)

## Улучшение читаемости датасета
# группируем так, чтобы каждому blend_id соответствовала одна строка. сохраняем только свойства относящиеся к маслу, включая id компонент и smiles

if verbose:
    print("Предобработка датасета")
    
def oil_param_finder(all_oil_info, param_title):
    '''нахождение параметра масла по id_параметра'''
    props = all_oil_info[all_oil_info["oil_property_param_title"] == param_title]
    if len(props) <= 0:
        return None
    if props["oil_property_param_value"].nunique() == 1:
        return props["oil_property_param_value"].iloc[0]
    else:
        # print(props["oil_property_param_value"])
        return "err"

df = df0.copy()
ids = df.blend_id.unique() #список уникальных blend_id
props = df.oil_property_param_title.unique() #список уникальных id свойств масла
temp = [] 
for id_ in ids:
    all_oil_info = df[df.blend_id == id_] #вся информация по маслу с определённым blend_id

    #определение типа масла
    oil_types = all_oil_info["oil_type"]
    if oil_types.nunique() == 1:
        oil_type = oil_types.iloc[0]
    else:
        # print(oil_types.unique())
        oil_type = "err"

    #определение свойств масла
    prop_vals = [oil_param_finder(all_oil_info, prop) for prop in props]

    #создание списка smiles входящих в масло
    components = all_oil_info.smiles.dropna().unique().tolist()

    #список id_компонент
    component_names = all_oil_info.component_name.unique().tolist()
    
    temp+=[[
        id_, 
        oil_type
    ] + prop_vals + [
        components,
        component_names
    ],]

oil_test = pd.DataFrame(temp, columns = ["blend_id", "oil_type"] + props.tolist() + ["smiles_list", "component_names_list"])


### Подготовка фингерпринтов для молекул
if verbose:
    print("Подготовка фингерпринтов для молекул")
    
all_smiles = oil_test.smiles_list.sum() #список всех уникальных smiles
unique_smiles, counts = np.unique(all_smiles, return_counts=True) #гисограмма по smiles
smiles_stat = pd.DataFrame({"smiles": unique_smiles, "count": counts}).sort_values(by="count", ascending=False)

temp = []
for smile in smiles_stat.smiles:
    temp += [[smile,] + np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))).tolist()]

smiles_fingerprints = pd.DataFrame(temp, columns = ["smiles"]+[f"key{i+1}" for i in range(len(temp[0])-1)])

null_keys = ['key1','key2','key3','key4','key5','key6','key7','key8','key10','key14','key15','key16','key17','key18','key20','key21','key22','key23','key25','key26','key27','key28','key29','key31','key32','key33','key34','key35','key36','key38','key39','key42','key43','key44','key46','key47','key48','key50','key55','key57','key58','key63','key64','key69','key72','key73','key79','key85','key88','key99','key120'] #ключи для которых на молекулах из тернировочного датасета все значения соответсвующих элементов фингерпринов равны нулю

smiles_fingerprints = smiles_fingerprints.drop(columns = null_keys) #итоговый DataFrame со smiles и очищенныи фингерпринтами


### Добавление свойств масла в таблицу
if verbose:
    print("Добавление свойств масла в таблицу")
    
## Добавление столбцов
needed_props = ['4c7a51f1-dc82-41dc-92fa-772535c2c70c', '33fd9876-db06-478c-8993-17dd5d9d698a']
df_oil_props = oil_test[["blend_id","smiles_list"]+needed_props]

## Заполнение пропущенных значений средними
df_oil_props["4c7a51f1-dc82-41dc-92fa-772535c2c70c"] = df_oil_props["4c7a51f1-dc82-41dc-92fa-772535c2c70c"].fillna(11.58893382352941)
df_oil_props["33fd9876-db06-478c-8993-17dd5d9d698a"] = df_oil_props["33fd9876-db06-478c-8993-17dd5d9d698a"].fillna(-47.74193548387097)

## Смена знака для столбца с отрицательными значениями
df_oil_props["33fd9876-db06-478c-8993-17dd5d9d698a"] = df_oil_props["33fd9876-db06-478c-8993-17dd5d9d698a"].apply(lambda x: -x)


### Добавление фингерпринтов в таблицу
#фингерпринт для смеси - сумма фингерпринтов всех смайлов входящих в смесь
if verbose:
    print("Добавление фингерпринтов")

def get_fngprint_from_df(df_fngprints, smiles):
    '''функция нахождения фингерпринта в виде list по smiles в датафрейме df_fngprints'''
    return df_fngprints[df_fngprints.smiles == smiles].drop(columns = ["smiles"]).iloc[0].tolist()

def sum_of_fngprints(smiles):
    '''функция нахождения фингерпринта смеси (суммы фингерпринтов всех составляющих)'''
    s = np.zeros(167-51)
    for smile in smiles:
        # print(smile)
        s += np.array(get_fngprint_from_df(smiles_fingerprints, smile))
    return s

df_oil_props = pd.concat([df_oil_props, df_oil_props.smiles_list.apply(sum_of_fngprints).apply(pd.Series)], axis=1)


### Добавление дескрипторов в таблицу
if verbose:
    print("Добавление дескрипторов")
    
df_oil_props['n_mol']=df_oil_props['smiles_list'].apply(n_mol)
df_oil_props['n_atom_mean']=df_oil_props['smiles_list'].apply(n_atom_mean)
df_oil_props['frac_atom_с']=df_oil_props['smiles_list'].apply(frac_atom_с)
df_oil_props['frac_atom_o']=df_oil_props['smiles_list'].apply(frac_atom_o)
df_oil_props['mol_w_mean']=df_oil_props['smiles_list'].apply(mol_w_mean)
df_oil_props['MolLogP_mean']=df_oil_props['smiles_list'].apply(MolLogP_mean)
df_oil_props['TPSA_sum']=df_oil_props['smiles_list'].apply(TPSA_sum)
df_oil_props['n_bonds_mean']=df_oil_props['smiles_list'].apply(n_bonds_mean)
df_oil_props['n_bonds_nn_mean']=df_oil_props['smiles_list'].apply(n_bonds_nn_mean)
df_oil_props['n_col']=df_oil_props['smiles_list'].apply(n_col)
df_oil_props['ind_V_mean']=df_oil_props['smiles_list'].apply(ind_V_mean)
df_oil_props['LabuteASA']=df_oil_props['smiles_list'].apply(LabuteASA)
df_oil_props['HallKierAlpha']=df_oil_props['smiles_list'].apply(HallKierAlpha)


### Приведение таблицы в вид использованный для обучения
df_oil_props.columns = [str(col) for col in df_oil_props.columns] #смена типа имени колонок на строки


### Предсказание
if verbose:
    print("Предсказание")

## Загрузка моделей
rfr = joblib.load("random_forest_regressor_model.pkl") #обченный RandomForestRegressor
scaler = joblib.load("scaler.pkl") #и скейлер для целевой переменной

## Сохранение результата
pd.DataFrame(
    {
        "blend_id": df_oil_props["blend_id"].to_list(), 
        "results": 10**(scaler.inverse_transform(rfr.predict(df_oil_props.drop(columns=["blend_id", "smiles_list"])).reshape(-1, 1)).reshape(1, -1)[0])
    }
).to_csv(output_file, index=False)
if verbose:
    print("Результат сохранён.")