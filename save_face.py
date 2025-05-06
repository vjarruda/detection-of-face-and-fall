import face_recognition
import cv2
import pickle

nome = input("Digite o nome da pessoa a ser cadastrada: ")

video_capture = cv2.VideoCapture(0)

print("Pressione 's' para capturar e salvar o rosto.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    cv2.imshow("Captura - Pressione 's' para salvar", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        #conversao necessaria
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #verifica rostos
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            print("Nenhum rosto detectado. Tente novamente.")
            continue
        #codifica o rosto
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            with open(f"{nome}.pkl", "wb") as f:
                pickle.dump(face_encodings[0], f)
            print(f"Rosto de '{nome}' salvo com sucesso.")
        else:
            print("Erro ao codificar o rosto.")

        break

video_capture.release()
cv2.destroyAllWindows()