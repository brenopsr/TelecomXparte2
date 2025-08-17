# TelecomX – Parte 2: Modelagem Preditiva de Churn

Esta segunda etapa do desafio Telecom X dá sequência ao trabalho iniciado na análise exploratória.  
O objetivo é construir modelos de machine learning capazes de antecipar quais clientes têm maior probabilidade de cancelar o serviço (churn).  

A equipe de Machine Learning da Telecom X espera que você entregue um **pipeline completo de pré-processamento, modelagem, avaliação e interpretação dos resultados**.  

O arquivo de dados utilizado é o mesmo da parte 1, já tratado e limpo, contendo **7.267 registros e 21 variáveis** após remoção de identificadores.

---

## 1. Preparação dos dados

### 1.1 Carregamento e limpeza
- O conjunto de dados foi carregado a partir de `telecomx_clean.csv` (gerado na parte 1).  
- Colunas irrelevantes (identificadores) foram removidas.  
- A variável-alvo `Churn` foi convertida em binária (1 para clientes que cancelaram, 0 para os demais).  
- Dataset final: **7.267 registros e 21 colunas** (incluindo `Churn`).

### 1.2 Tipos de variáveis e codificação
- Numéricas: `customer_tenure`, `account_Charges_Monthly`, `account_Charges_Total`, `daily_charge`.  
- Categóricas: `account_Contract`, `internet_InternetService`, `account_PaymentMethod` e variáveis demográficas.  
- Codificação: **one-hot encoding** aplicada às categóricas.  
- Normalização (`StandardScaler`) aplicada apenas para modelos sensíveis à escala (LR e KNN).  

### 1.3 Proporção de churn e desbalanceamento
- 27,4% dos clientes cancelaram (Churn = 1).  
- 72,6% permaneceram.  
- Não é um desbalanceamento extremo, mas foi monitorado.  
- Técnicas de oversampling (SMOTE) foram testadas, mas não adicionaram ganhos significativos.

### 1.4 Distribuição e correlação
- Distribuição das classes (Churn 0 e 1):  
  <Figure size 640x480 with 1 Axes><img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/0b90d07e-aa21-4c71-9cdb-66b52d0605ed" />


- Matriz de correlação (heatmap):  
  <Figure size 800x600 with 2 Axes><img width="687" height="590" alt="image" src="https://github.com/user-attachments/assets/3513bf98-0a0f-4076-b602-bc7f25969f41" />


- Boxplots de variáveis relevantes (`customer_tenure`, `account_Charges_Total`, `daily_charge`):  
  <Figure size 640x480 with 1 Axes><img width="630" height="455" alt="image" src="https://github.com/user-attachments/assets/952b201c-988f-4ad7-9692-10e9cae23dac" />

  <Figure size 640x480 with 1 Axes><img width="630" height="455" alt="image" src="https://github.com/user-attachments/assets/afe9fee7-c231-48b2-8ce7-50dfe65d3072" />

  <Figure size 640x480 with 1 Axes><img width="630" height="455" alt="image" src="https://github.com/user-attachments/assets/e418ed83-d183-4ac0-ab4f-96a7637e5cd6" />




Principais achados:
- Clientes com contratos mais curtos tendem a cancelar mais.  
- Gastos totais e diários mais altos estão associados a maior probabilidade de churn.  

---

## 2. Divisão de dados e criação dos modelos

- Dados divididos em **80% treino** e **20% teste**, estratificando o Churn.  
- Modelos treinados:  
  - **Regressão Logística (LR)** – requer normalização.  
  - **KNN (k=15)** – baseado em distância, normalizado.  
  - **Random Forest (RF)** – 300 e 400 árvores, sem necessidade de normalização.  
- GridSearchCV aplicado para ajuste de hiperparâmetros (C na LR, número de árvores e profundidade na RF).  

---

## 3. Avaliação e comparação de modelos

Métricas: **Acurácia, Precisão, Recall, F1-score, AUC** + Matriz de Confusão.  

| Modelo                        | Acurácia | Precisão | Recall | F1   | AUC  |
|-------------------------------|----------|----------|--------|------|------|
| Regressão Logística            | 0.803    | 0.636    | 0.543  | 0.586| 0.844|
| Random Forest                  | 0.779    | 0.584    | 0.481  | 0.528| 0.819|
| KNN                            | 0.790    | 0.594    | 0.575  | 0.584| 0.817|
| Regressão Logística (Tuning)   | 0.802    | 0.635    | 0.540  | 0.584| 0.844|
| Random Forest (Tuning)         | 0.792    | 0.623    | 0.487  | 0.547| 0.842|

- Matrizes de confusão:  
<Figure size 640x480 with 2 Axes><img width="527" height="470" alt="image" src="https://github.com/user-attachments/assets/07923fba-3515-4c59-aca8-cacf96c9360c" />

<Figure size 640x480 with 2 Axes><img width="527" height="470" alt="image" src="https://github.com/user-attachments/assets/4ddb98bc-fce1-4e4d-8ddc-cea3776a1e21" />

<Figure size 640x480 with 2 Axes><img width="527" height="470" alt="image" src="https://github.com/user-attachments/assets/7780d432-4ce6-4753-9e92-0f5db2ae7039" />

- Curvas ROC:  
<Figure size 640x480 with 1 Axes><img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/bd97b186-9a83-4ba2-9a6a-8ef77b9115bc" />

<Figure size 640x480 with 1 Axes><img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/db2522f4-9bd7-46c1-a3a2-d57e86d98fac" />

<Figure size 640x480 with 1 Axes><img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/753839f8-e1a6-4564-b574-6186811fa44a" />
  
Destaques:
- LR obteve maior **AUC (~0,84)**.  
- RF teve melhor interpretabilidade.  
- KNN conseguiu recall levemente superior.  
- Tuning não trouxe ganhos significativos.  

---

## 4. Importância das variáveis

- Importância das variáveis na RF:  
  <Figure size 1000x500 with 1 Axes><img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/50b93520-88dd-45fe-8f0f-de1064978a31" />


- Coeficientes da LR (absolutos):  
  <Figure size 1000x500 with 1 Axes><img width="976" height="490" alt="image" src="https://github.com/user-attachments/assets/d20de114-12a6-4630-87cd-e0342c8b44ee" />


Insights:
- **Tempo de contrato**: mais longo → menor churn.  
- **Gastos totais/diários/mensais**: maiores → maior churn.  
- **Tipo de contrato**: `month-to-month` → maior risco.  
- **Internet Fibra**: maior propensão ao churn.  
- **Pagamento via Electronic Check**: mais evasão.  
- Demografia (idade > 65, gênero feminino) aparece com menor peso.  

---

## 5. Conclusões e recomendações

- **Modelo preferido**: Regressão Logística → melhor AUC e desempenho consistente.  
- **RF** → boa interpretabilidade.  
- **KNN** → recall superior, mas menos prático em escala.  

Recomendações estratégicas:
1. **Migrar clientes de contratos mensais** para planos anuais/bianuais.  
2. **Monitorar clientes novos** (primeiros meses críticos).  
3. **Oferecer planos personalizados** para clientes de alto gasto.  
4. **Incentivar formas automáticas de pagamento** (débito/cartão).  

---

📌 Em resumo:  
A combinação **Regressão Logística + Random Forest** fornece previsões robustas e insights estratégicos.  
O próximo passo é integrar o modelo no CRM da TelecomX e lançar campanhas de retenção focadas.  

---

