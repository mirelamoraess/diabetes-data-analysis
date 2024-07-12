from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carregar o dataset de diabetes
dataset_diabetes = load_diabetes()
print(dataset_diabetes.feature_names)
print(dataset_diabetes.target)

# Dividir o dataset em treinamento e teste
X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(
   dataset_diabetes.data, dataset_diabetes.target, stratify=None, random_state=42)

training_accuracy = []
test_accuracy = []

# Treinar e avaliar o modelo com e sem interceptação
for interception in [True, False]:
   regr = LinearRegression(fit_intercept=interception)
   regr.fit(X_train_dia, y_train_dia)
   training_accuracy.append(regr.score(X_train_dia, y_train_dia))
   test_accuracy.append(regr.score(X_test_dia, y_test_dia))
   
# Plotar os resultados
plt.plot(["Interc", "No Interc"], training_accuracy, label='Acurácia no conj. treino')
plt.plot(["Interc", "No Interc"], test_accuracy, label='Acurácia no conj. teste')
plt.ylabel('Acurácia')
plt.xlabel('Fit Intercept')
plt.legend()
plt.show()