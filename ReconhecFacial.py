import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inicializa o classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para capturar fotos e salvar em uma pasta
def capturar_fotos(nome):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduz a resolução da câmera
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    contador = 0
    pasta = f"D:\\imagens\\datasets\\LIVE\\fotos\\/{nome}"

    if not os.path.exists(pasta):
        os.makedirs(pasta)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        cv2.imshow("Captura de Fotos", frame)

        # Salva a foto a cada 5 frames
        if contador % 5 == 0:
            caminho_foto = os.path.join(pasta, f"{nome}_{contador}.jpg")
            cv2.imwrite(caminho_foto, frame)

        contador += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Adiciona um delay entre as capturas
        cv2.waitKey(100)

    camera.release()
    cv2.destroyAllWindows()

# Função para extrair embeddings usando Haar Cascade
def extract_embeddings(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        return face.flatten()
    return None

# Função para treinar o modelo
def treinar_modelo():
    data = []
    labels = []
    for person_name in os.listdir('D:\\imagens\\datasets\\LIVE\\fotos\\'):
        person_dir = os.path.join('D:\\imagens\\datasets\\LIVE\\fotos\\', person_name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            embedding = extract_embeddings(image_path)
            if embedding is not None:
                data.append(embedding)
                labels.append(person_name)
    data = np.array(data)
    labels = np.array(labels)

    if len(set(labels)) < 2:
        raise ValueError("O número de classes precisa ser maior que um. Certifique-se de que há imagens de pelo menos duas pessoas diferentes.")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    model = SVC(kernel='linear', probability=True)
    model.fit(data, labels)

    # Avaliação do modelo
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    return model, label_encoder

# Função para reconhecimento facial em tempo real
def reconhecimento_facial(model, label_encoder):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduz a resolução da câmera
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face_array = face.flatten().reshape(1, -1)
            probabilities = model.predict_proba(face_array)[0]
            best_match_index = np.argmax(probabilities)
            best_match_probability = probabilities[best_match_index]
            if best_match_probability < 0.75:
                name = "unknown"
            else:
                name = label_encoder.inverse_transform([best_match_index])[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Reconhecimento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Interface gráfica com Tkinter
def iniciar_interface():
    root = tk.Tk()
    root.geometry("300x200")
    root.title("Sistema de Reconhecimento Facial")

    def novo_cadastro():
        nome = simpledialog.askstring("Cadastro", "Digite o nome da pessoa:")
        if nome:
            capturar_fotos(nome)
            tk.messagebox.showinfo("Cadastro", f"Fotos de {nome} capturadas com sucesso!")

    def iniciar_reconhecimento():
        model, label_encoder = treinar_modelo()
        reconhecimento_facial(model, label_encoder)

    btn_cadastro = tk.Button(root, text="Novo Cadastro", command=novo_cadastro)
    btn_cadastro.pack(pady=10)

    btn_reconhecimento = tk.Button(root, text="Iniciar Reconhecimento", command=iniciar_reconhecimento)
    btn_reconhecimento.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    iniciar_interface()