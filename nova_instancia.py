import warnings
warnings.filterwarnings("ignore")
from pickle import load
import numpy as np

# Carregar o modelo e o normalizador
crop_tree = load(open('shape_tree_model_cross.pkl', 'rb'))
shape_normalizador = load(open('shape_normalizador.pkl', 'rb'))

# Função para classificar uma nova instância de dados
def classificar_instancia(nova_instancia):
    nova_instancia_normalizada = shape_normalizador.transform([nova_instancia])
    classe_predita = crop_tree.predict(nova_instancia_normalizada)
    probabilidades = crop_tree.predict_proba(nova_instancia_normalizada)
    return classe_predita[0], probabilidades

# Função para imprimir resultados dentro de uma "caixinha"
def imprimir_resultado(classe, scores):
    print("+" + "-"*38 + "+")
    print("|{:^38}|".format("Resultado da Classificação"))
    print("+" + "-"*38 + "+")
    print("|{:^38}|".format(f"Classe predita: {classe}"))
    print("+" + "-"*38 + "+")
    print("|{:^38}|".format("Scores de Classes:"))
    for idx, score in enumerate(scores[0]):
        print("|{:^38}|".format(f"Classe {idx+1}: {score:.4f}"))
    print("+" + "-"*38 + "+")

# Exemplo de uso da função de classificação
def main():
    # Nova instância para teste
    nova_instancia = [28395, 610.291, 208.1781167, 173.888747, 1.197191424, 0.549812187, 28715, 190.1410973, 0.763922518, 0.988855999, 0.958027126, 0.913357755, 0.007331506, 0.003147289, 0.834222388, 0.998723889]
    classe, scores = classificar_instancia(nova_instancia)
    imprimir_resultado(classe, scores)
    
    # Outra instância para teste
    nova_instancia = [46704,813.324,295.3600895,203.0207835,1.454826863,0.726310254,47182,243.8552433,0.701735407,0.989869018,0.887231119,0.825620157,0.006324086,0.001812586,0.681648643,0.991680757]
    classe, scores = classificar_instancia(nova_instancia)
    imprimir_resultado(classe, scores)

if __name__ == "__main__":
    main()
