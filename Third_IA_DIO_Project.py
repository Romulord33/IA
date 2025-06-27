# Importando bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import io


# Verificando a versão do TensorFlow instalada
print(tf.__version__)

# Carregando o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizando e ajustando o formato das imagens
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Definindo o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback do TensorBoard
logdir = 'logs/images'
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

# Treinando o modelo
model.fit(train_images, train_labels, epochs=5,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

# Fazendo previsões
y_pred = np.argmax(model.predict(test_images), axis=-1)

# Gerando a matriz de confusão
con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_curve, auc

# Funções para métricas de avaliação
def calcular_metrica(conf_mat, classe):
    """Calcula VP, VN, FP, FN para uma classe específica"""
    VP = conf_mat[classe, classe]
    FP = sum(conf_mat[:, classe]) - VP
    FN = sum(conf_mat[classe, :]) - VP
    VN = conf_mat.sum() - (VP + FP + FN)
    return VP, VN, FP, FN

def calcular_acuracia(VP, VN, FP, FN):
    return (VP + VN) / (VP + VN + FP + FN)

def calcular_sensibilidade(VP, FN):
    return VP / (VP + FN)

def calcular_especificidade(VN, FP):
    return VN / (VN + FP)

def calcular_precisao(VP, FP):
    return VP / (VP + FP)

def calcular_fscore(precisao, sensibilidade):
    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

# Escolhendo uma classe para o cálculo das métricas (binário, ex.: classe 0)
classe_escolhida = 0
VP, VN, FP, FN = calcular_metrica(con_mat, classe_escolhida)

# Calculando as métricas
acuracia = calcular_acuracia(VP, VN, FP, FN)
sensibilidade = calcular_sensibilidade(VP, FN)
especificidade = calcular_especificidade(VN, FP)
precisao = calcular_precisao(VP, FP)
fscore = calcular_fscore(precisao, sensibilidade)

# Exibindo os resultados
print(f"Para a classe {classe_escolhida}:")
print(f"Acurácia: {acuracia:.2f}")
print(f"Sensibilidade (Recall): {sensibilidade:.2f}")
print(f"Especificidade: {especificidade:.2f}")
print(f"Precisão: {precisao:.2f}")
print(f"F-Score: {fscore:.2f}")

# Curva ROC
# Calculando scores binarizados para uma classe específica
y_true_bin = (test_labels == classe_escolhida).astype(int)
y_pred_prob = model.predict(test_images)[:, classe_escolhida]

# Calculando a curva ROC
fpr, tpr, _ = roc_curve(y_true_bin, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotando a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()
