import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Abre um diálogo para selecionar uma imagem
Tk().withdraw()  # Evita que a janela principal do Tkinter apareça
filename = askopenfilename(title="Selecione uma imagem", filetypes=[("Arquivos de imagem", "*.jpg;*.jpeg;*.png;*.bmp")])

# Verifica se o usuário selecionou um arquivo
if filename:
    # Carrega a imagem selecionada
    img = cv2.imread(filename)

    # Verifica se a imagem foi carregada corretamente
    if img is None:
        print("Erro: A imagem não pôde ser carregada. Verifique o caminho do arquivo.")
    else:
        # Converte a imagem para tons de cinza
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplica o filtro de desfoque gaussiano para suavizar a imagem
        suave = cv2.GaussianBlur(img_gray, (7, 7), 0)

        # Aplica a binarização usando um limiar de 160
        (T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)

        # Aplica a binarização inversa usando um limiar de 160
        (T, bin_inv) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)

        # Junta as imagens para exibição
        resultado = np.vstack([
            np.hstack([img, cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)]),
            np.hstack([cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bin_inv, cv2.COLOR_GRAY2BGR)])
        ])

        # Exibe o resultado em uma janela
        cv2.imshow("Imagem Original e Transformações", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Nenhuma imagem foi selecionada.")
