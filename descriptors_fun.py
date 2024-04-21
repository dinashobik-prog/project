from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments #для создания дескрипторов по smiles

# функции создания новых предикторов:

# 1. количество компонентов в смеси:
def n_mol(smiles):
  return len(smiles)

# 2. средний размер молекулы (количество атомов)
def n_atom_mean(smiles):
  N=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      N+=mol.GetNumAtoms()
    N/=len(smiles)
  return N

# 3. Доля атомов С:
def frac_atom_с(smiles):
  N=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      N+=sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    N/=len(smiles)
  return N
# 4. Доля атомов O:
def frac_atom_o(smiles):
  N=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      N+=sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    N/=len(smiles)
  return N
# 5. Средний молекулярный вес:
def mol_w_mean(smiles):
  w=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      w+=Descriptors.ExactMolWt(mol)
    w/=len(smiles)
  return w
# 6. Среднее значение логарифма коэффициента распределения:
def MolLogP_mean(smiles):
  m=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      m+=Descriptors.MolLogP(mol)
    m/=len(smiles)
  return m
# 7. Общая полярная поверхность: сумма полярных поверхностей всех компонентов:
def TPSA_sum(smiles):
  TPSA=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      TPSA+=Descriptors.TPSA(mol)
  return TPSA
# 8. Cреднее количество связей:
def n_bonds_mean(smiles):
  n=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      n+=mol.GetNumBonds()
    n/=len(smiles)
  return n
# 9. Средняя степень насыщенности:
def n_bonds_nn_mean(smiles):
  n=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      n+=sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
      n+=sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)
    n/=len(smiles)
  return n
# 10. Общее количество колец:
def n_col(smiles):
  n=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      n+=mol.GetRingInfo().NumRings()
    n/=len(smiles)
  return n
# 11. Средние топологические индексы (Винера):
def ind_V_mean(smiles):
  i=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
      i+=0.5*sum(sum(distance_matrix[i, j] for j in range(i + 1, mol.GetNumAtoms())) for i in range(mol.GetNumAtoms()))
    i/=len(smiles)
  return i
# 12. Площадь доступной поверхности, вычисленная по методу Лабута (гидрофобность):
def LabuteASA(smiles):
  n=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      n+=Descriptors.LabuteASA(mol)
  return n
# 13. Доп:
def HallKierAlpha(smiles):
  n=0
  if len(smiles)!=0:
    for el in smiles:
      mol = Chem.MolFromSmiles(el)
      n+=Descriptors.HallKierAlpha(mol)
    n/=len(smiles)
  return n