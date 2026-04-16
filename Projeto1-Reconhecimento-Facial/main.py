import cv2
import numpy as np
import os
import urllib.request
import ctypes

class SistemaFacial:
    def __init__(self):
        # 1. Ajuste de Caminho para Windows (evita erro de acentos)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.name == 'nt':
            try:
                buf = ctypes.create_unicode_buffer(1024)
                ctypes.windll.kernel32.GetShortPathNameW(base_dir, buf, 1024)
                self.base_dir = buf.value
            except:
                self.base_dir = base_dir
        else:
            self.base_dir = base_dir

        # 2. Configurações de pastas
        self.path_fotos = os.path.join(self.base_dir, 'data', 'fotos')
        self.path_modelos = os.path.join(self.base_dir, 'data', 'modelos')
        self.path_xml = os.path.join(self.base_dir, 'classificadores')
        self.largura, self.altura = 220, 220
        
        # Índice da câmera padrão (0 é geralmente a do notebook)
        self.id_camera_atual = 0 

        for p in [self.path_fotos, self.path_modelos, self.path_xml]:
            os.makedirs(p, exist_ok=True)

        self.xml_face = os.path.join(self.path_xml, 'haarcascade_frontalface_default.xml')
        self.xml_eye = os.path.join(self.path_xml, 'haarcascade_eye.xml')
        self._garantir_xmls()

        self.face_cascade = cv2.CascadeClassifier(self.xml_face)
        self.eye_cascade = cv2.CascadeClassifier(self.xml_eye)

    def _garantir_xmls(self):
        urls = {
            self.xml_face: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            self.xml_eye: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
        }
        for path, url in urls.items():
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)

    def alterar_camera(self):
        """Permite ao usuário escolher entre câmera 0, 1 ou 2."""
        print(f"\nCâmera atual: {self.id_camera_atual}")
        novo_id = input("Digite o índice da câmera (0 para interna, 1 ou 2 para USB): ")
        if novo_id.isdigit():
            self.id_camera_atual = int(novo_id)
            print(f"Câmera alterada para {self.id_camera_atual}!")
        else:
            print("Entrada inválida.")

    def _pre_processar(self, frame_cinza):
        """Melhora contraste para peles negras usando CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(frame_cinza)

    def capturar_fotos(self):
        id_usuario = input('ID numérico: ').strip()
        camera = cv2.VideoCapture(self.id_camera_atual)
        
        if not camera.isOpened():
            print(f"Erro: Não foi possível abrir a câmera {self.id_camera_atual}.")
            return

        amostra = 1
        max_amostras = 30
        print(f"Capturando... Olhe para a câmera {self.id_camera_atual}. 'Q' para sair.")

        while True:
            sucesso, frame = camera.read()
            if not sucesso: break
            
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cinza_otimizado = self._pre_processar(cinza)
            
            faces = self.face_cascade.detectMultiScale(cinza_otimizado, 1.1, 4, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 2)
                roi_cinza = cinza_otimizado[y:y+h, x:x+w]
                
                if amostra <= max_amostras:
                    face_img = cv2.resize(roi_cinza, (self.largura, self.altura))
                    cv2.imwrite(os.path.join(self.path_fotos, f"pessoa.{id_usuario}.{amostra}.jpg"), face_img)
                    print(f"Foto {amostra} capturada.")
                    amostra += 1

            cv2.imshow("Captura - Pressione Q", frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (amostra > max_amostras):
                break

        camera.release()
        cv2.destroyAllWindows()

    def treinar(self):
        arquivos = [f for f in os.listdir(self.path_fotos) if f.endswith('.jpg')]
        faces, ids = [], []

        if not arquivos:
            print("Sem fotos para treinar."); return

        for arquivo in arquivos:
            caminho = os.path.join(self.path_fotos, arquivo)
            faces.append(cv2.imread(caminho, cv2.IMREAD_GRAYSCALE))
            ids.append(int(arquivo.split('.')[1]))

        print("Treinando... aguarde.")
        reconhecedor = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        reconhecedor.train(faces, np.array(ids))
        reconhecedor.write(os.path.join(self.path_modelos, 'classificadorLBPH.yml'))
        print("Treino finalizado com sucesso!")

    def reconhecer(self):
        modelo = os.path.join(self.path_modelos, 'classificadorLBPH.yml')
        if not os.path.exists(modelo):
            print("Treine o sistema primeiro."); return

        reconhecedor = cv2.face.LBPHFaceRecognizer_create()
        reconhecedor.read(modelo)
        camera = cv2.VideoCapture(self.id_camera_atual)

        while True:
            sucesso, frame = camera.read()
            if not sucesso: break
            
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cinza_otimizado = self._pre_processar(cinza)
            faces = self.face_cascade.detectMultiScale(cinza_otimizado, 1.1, 5)

            for (x, y, w, h) in faces:
                roi_face = cv2.resize(cinza_otimizado[y:y+h, x:x+w], (self.largura, self.altura))
                id_usuario, confianca = reconhecedor.predict(roi_face)
                
                cor = (0, 255, 0) if confianca < 75 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
                cv2.putText(frame, f"ID {id_usuario} ({round(confianca, 1)})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

            cv2.imshow("Reconhecimento - Q para sair", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        camera.release()
        cv2.destroyAllWindows()

# --- MENU PRINCIPAL ---
if __name__ == "__main__":
    app = SistemaFacial()
    while True:
        print(f"\n[ Câmera Atual: {app.id_camera_atual} ]")
        print("1. Capturar | 2. Treinar | 3. Reconhecer | 4. Alternar Câmera | 0. Sair")
        op = input("Opção: ")
        
        match op:
            case '1': app.capturar_fotos()
            case '2': app.treinar()
            case '3': app.reconhecer()
            case '4': app.alterar_camera()
            case '0': break
            case _: print("Opção inválida.")
            