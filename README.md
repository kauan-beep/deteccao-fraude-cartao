# Detecção de fraude em cartão de crédito

Projeto da DIO (Detecção de Anomalias em Transações em Python): notebook que treina e compara modelos para detectar fraude numa base em que só 0,17% das transações são fraude.

## Arquivos

- `deteccao_fraude_cartao.ipynb`: notebook completo, com as saídas salvas (tabelas e gráficos). As bibliotecas são instaladas na primeira célula.

O dataset não está no repositório: o notebook carrega pelo link `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` (variável `DATA_URL`).

## 1. O problema

A base tem 284.807 transações, com as colunas `Time`, `Amount`, `Class` e `V1` a `V28` (variáveis anonimizadas por PCA). Só 492 são fraude (0,1727%). Um modelo que responde "não é fraude" para tudo teria 99,8273% de acurácia e recall da fraude igual a 0. Por isso a acurácia não serve aqui. A métrica principal é o **recall da fraude**, acompanhado de precisão, F1 e PR-AUC.

## 2. Preparação dos dados

- Remoção de 1.081 linhas duplicadas.
- Variável nova `Amount_log = log1p(Amount)`.
- `StandardScaler` em `Time` e `Amount_log`, ajustado só no treino.
- Divisão estratificada, mantendo ~0,17% de fraude em todas as partes: treino 181.584 linhas (302 fraudes), validação 45.396 (76 fraudes) e teste 56.746 (95 fraudes).
- Desbalanceamento tratado com peso de classes: `class_weight` na Regressão Logística e no Random Forest, `scale_pos_weight` = 600,3 no XGBoost.

## 3. Comparação dos modelos (classe fraude, conjunto de teste)

Limiar padrão 0,5:

| Modelo | Recall | Precisão | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,8737 | 0,0509 | 0,0962 | 0,6785 | 0,9586 |
| Random Forest | 0,7158 | 0,8831 | 0,7907 | 0,7840 | 0,9625 |
| XGBoost | 0,7474 | 0,9103 | 0,8208 | 0,8155 | 0,9709 |

Limiar escolhido na validação (maior precisão com recall mínimo de 0,85):

| Modelo | Limiar | Recall | Precisão | F1 |
|---|---|---|---|---|
| Regressão Logística | 0,9999 | 0,7684 | 0,7849 | 0,7766 |
| Random Forest | 0,4192 | 0,7368 | 0,8861 | 0,8046 |
| XGBoost | 0,9163 | 0,7474 | 0,9861 | 0,8503 |

A Regressão Logística tem o maior recall em 0,5, mas com precisão de 5%, o que gera muitos falsos alarmes (cerca de 1.500, estimados a partir da precisão e do recall). O XGBoost teve o melhor F1, PR-AUC e ROC-AUC entre os três.

Reamostragem (XGBoost treinado em dados reamostrados só no treino, limiar 0,5):

| Estratégia | Recall | Precisão | F1 | PR-AUC |
|---|---|---|---|---|
| Undersampling | 0,8737 | 0,0313 | 0,0604 | 0,6598 |
| SMOTE (oversampling) | 0,7895 | 0,7576 | 0,7732 | 0,8083 |
| Peso de classes (sem reamostrar) | 0,7474 | 0,9103 | 0,8208 | 0,8155 |

O undersampling manteve o recall alto, mas com precisão de 3%. O SMOTE detectou mais fraudes que o peso de classes (recall 0,7895 contra 0,7474), com menos precisão e F1 menor.

## 4. Limiar de decisão e SHAP

**Limiar:** o limiar do XGBoost foi escolhido na validação (maior precisão com recall de pelo menos 0,85) e ficou em **0,9163**. No teste: recall 0,7474 (71 das 95 fraudes detectadas), precisão 0,9861 e F1 0,8503. A matriz de confusão tem 56.650 normais corretos, 1 falso positivo, 24 fraudes perdidas e 71 detectadas. Em relação ao limiar 0,5, o recall foi o mesmo (71 fraudes), mas os falsos positivos caíram de 7 para 1. A meta de recall de 0,85 usada na validação não foi atingida no teste.

**SHAP:** a variável mais importante foi `V14`, seguida de `V4`, tanto na importância do XGBoost quanto no gráfico SHAP. No gráfico, valores baixos de `V14` empurram a previsão para fraude e valores altos de `V4` também. Na transação individual explicada, a probabilidade de fraude foi 0,9980: `V14 = -3,823` contribuiu +5,89, `V10 = -3,539` contribuiu +2,28, `V12 = -3,993` contribuiu +1,7 e `V4 = 2,213` contribuiu +0,93, enquanto `V28` puxou para baixo (-1,81). Como `V1` a `V28` são anônimas (PCA), não é possível dizer o que cada uma representa. `Amount_log` teve peso pequeno.

## 5. Limitações

- O teste tem só 95 fraudes: errar ou acertar um caso muda o recall em cerca de 1 ponto percentual.
- O modelo final foi escolhido pelo maior F1 no conjunto de teste, o que mistura escolha e avaliação. O ideal seria escolher pela validação.
- O limiar foi ajustado numa validação com só 76 fraudes, então é instável.
- A reamostragem foi avaliada no limiar 0,5, sem ajuste de limiar, então a comparação com o XGBoost de limiar ajustado não é direta.
- Foi usada uma única divisão dos dados e uma única semente, sem validação cruzada.

## 6. O que o notebook inclui além do roteiro mínimo

Segui o roteiro do desafio (explorar, preparar, treinar, avaliar e explicar) e, além do mínimo pedido, o notebook inclui:

- escolha do limiar numa validação separada do teste;
- remoção de linhas duplicadas antes da divisão dos dados;
- comparação de undersampling e SMOTE com o XGBoost de peso de classes;
- explicação de uma transação individual com o gráfico waterfall do SHAP, além do gráfico geral (beeswarm).
