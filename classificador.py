import warnings
warnings.filterwarnings("ignore")
import regex

# Abrir os dados
import pandas as pd

# Ler o arquivo CSV
dados = pd.read_csv('Dry_Bean_Dataset.csv', sep=';')

# Segmentar os dados em atributos e classes
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']

# Aplicar técnicas de pré-processamento (normalização)
dados_atributos = dados_atributos.replace({',':'.'}, regex=True).astype(float)

# 1 - Normalizar os dados
from sklearn.preprocessing import MinMaxScaler # Classe normalizadora

# Gerar o modelo normalizador para uso posterior
normalizador = MinMaxScaler()
shape_normalizador = normalizador.fit(dados_atributos) # o método fit() gera o modelo para normalização

# Salvar o modelo normalizador para uso posterior
from pickle import dump
dump(shape_normalizador, open('shape_normalizador.pkl', 'wb'))

# Normalizar a base de dados para treinamento
dados_atributos_normalizados = normalizador.fit_transform(dados_atributos) # o método fit_transform gera os dados normalizados

# Recompor os dados na forma de data frames
dados_finais = pd.DataFrame(dados_atributos_normalizados, columns=dados_atributos.columns)
dados_finais = dados_finais.join(dados_classes, how='left')

# 2. Balancear os dados
# Frequencia de classes conforme os dados originais
print('Freq das classes original: ', dados_classes.value_counts())

# Aplicar SMOTE para balanceamento dos dados
from imblearn.over_sampling import SMOTE
dados_atributos = dados_finais.drop(columns=['Class'])
dados_classes = dados_finais['Class']

# Construir um objeto a partir do SMOTE
resampler = SMOTE()
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

# Verificar a frequência das classes após o balanceamento
print('Frequência de classes após balanceamento')
from collections import Counter
classes_count = Counter(dados_classes_b)
print(classes_count)

# Converter os dados balanceados em DataFrames
dados_atributos_b = pd.DataFrame(dados_atributos_b, columns=dados_atributos.columns)
dados_classes_b = pd.DataFrame(dados_classes_b, columns=['Class'])

# Grid search para otimização de hiperparâmetros
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier()

# Montagem da grade de parâmetros
max_features = ['sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Iniciar a busca pelos melhores hiperparâmetros
rf_grid = GridSearchCV(tree, random_grid, refit=True, verbose=2)
rf_grid.fit(dados_atributos_b, dados_classes_b)

# Obter os melhores parâmetros
best_params = rf_grid.best_params_
print("##### MELHORES HIPERPARÂMETROS #####")
print(best_params)

# 3. Treinar o modelo

# Segmentar os dados em conjunto para treinamento e conjunto para testes (Test HoldOut)
from sklearn.model_selection import train_test_split
atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classes_b, test_size=0.3, random_state=42)

# Treinar o modelo
tree = DecisionTreeClassifier(**best_params)
crop_tree = tree.fit(atributos_train, classes_train)

# Pretestar o modelo
classes_test_predict = crop_tree.predict(atributos_test)

# AVALIAÇÃO DA ACURÁCIA COM CROSS VALIDATION
from sklearn.model_selection import cross_validate, cross_val_score
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree, dados_atributos_b, dados_classes_b, cv=10, scoring=scoring)

print('Precision:', scores_cross['test_precision_macro'].mean())
print('Recall:', scores_cross['test_recall_macro'].mean())
score_cross_val = cross_val_score(tree, dados_atributos_b, dados_classes_b, cv=10)
print('Cross-val score:', score_cross_val.mean(), ' - ', score_cross_val.std())

# Treinar o modelo com a base normalizada, balanceada e completa
crop_tree = tree.fit(dados_atributos_b, dados_classes_b)

# Salvar o modelo para uso posterior
dump(crop_tree, open('shape_tree_model_cross.pkl', 'wb'))

# Acurácia global do modelo
from sklearn import metrics
print('Acurácia global (provisória): ', metrics.accuracy_score(classes_test, classes_test_predict))

# MATRIZ DE CONFUSÃO
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Redesenha a matriz de confusão no novo tamanho
ConfusionMatrixDisplay.from_estimator(crop_tree, atributos_test, classes_test)

# Ajusta a rotação dos rótulos das classes no eixo x para melhor legibilidade
plt.xticks(rotation=90, ha='right')

# Exibe a matriz de confusão
plt.show()

# Implementar uma funcionalidade capaz de avaliar (classificar) uma nova instância de dados
# Está no código separado chamado `nova_instancia.py`