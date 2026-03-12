# Importe de bibliotecas
import sys
import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QGridLayout, QFileDialog, QMessageBox, QHBoxLayout
)

class AplicacionDeteccion(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Detección de Operaciones")
        
        # Variables de estado
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_frame)
        self.fotograma_actual = None

        self.iniciar_interfaz()

    def iniciar_interfaz(self):
        layout_principal = QGridLayout(self)

        # 1. Panel Izquierdo: Origen (Cámara o Imagen)
        self.lbl_origen = QLabel("Esperando origen de imagen...")
        self.lbl_origen.setFixedSize(500, 500)
        self.lbl_origen.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. Panel Derecho: Procesamiento (Contornos)
        self.lbl_procesado = QLabel("Esperando captura...")
        self.lbl_procesado.setFixedSize(500, 500)
        self.lbl_procesado.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 3. Controles
        layout_controles = QHBoxLayout()
        
        self.btn_subir = QPushButton("Subir Imagen")
        self.btn_subir.clicked.connect(self.cargar_imagen)
        
        self.btn_camara = QPushButton("Iniciar Cámara")
        self.btn_camara.clicked.connect(self.iniciar_camara_inteligente)
        
        self.btn_capturar = QPushButton("Tomar Fotografía")
        self.btn_capturar.clicked.connect(self.capturar_y_procesar)
        self.btn_capturar.setEnabled(False) # Se activa solo si hay cámara

        layout_controles.addWidget(self.btn_subir)
        layout_controles.addWidget(self.btn_camara)
        layout_controles.addWidget(self.btn_capturar)

        # Agregar al layout principal (sin estilos, formato default)
        layout_principal.addWidget(QLabel("Entrada Original"), 0, 0)
        layout_principal.addWidget(QLabel("Detección de Contornos"), 0, 1)
        layout_principal.addWidget(self.lbl_origen, 1, 0)
        layout_principal.addWidget(self.lbl_procesado, 1, 1)
        layout_principal.addLayout(layout_controles, 2, 0, 1, 2)

    # ******************************* Lógica de cámara y archivos *******************************
    def iniciar_camara_inteligente(self):
        """Inicializa la cámara buscando en múltiples índices."""
        if self.cap is not None:
            self.cap.release()

        orden = [2, 4, 1, 0, 3, 5]
        camara_ok = False

        for idx in orden:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    camara_ok = True
                    break
                else:
                    cap.release()

        if camara_ok:
            self.timer.start(30)
            self.btn_capturar.setEnabled(True)
        else:
            QMessageBox.critical(self, "Error", "No se encontró ninguna cámara.")

    def actualizar_frame(self):
        """Muestra el video en vivo en el panel izquierdo sin procesar."""
        if self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: return

        self.fotograma_actual = frame
        self.mostrar_en_label(frame, self.lbl_origen)

    def cargar_imagen(self):
        """Permite subir una imagen y detiene la cámara si está activa."""
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if ruta:
            if self.cap is not None:
                self.cap.release()
                self.timer.stop()
                self.btn_capturar.setEnabled(False)
                
            img = cv2.imread(ruta)
            if img is not None:
                self.fotograma_actual = img
                self.mostrar_en_label(img, self.lbl_origen)
                self.procesar_contornos(img) # Procesa automáticamente al subir

    def capturar_y_procesar(self):
        """Congela el fotograma actual de la cámara y ejecuta la visión artificial."""
        if self.fotograma_actual is not None:
            self.procesar_contornos(self.fotograma_actual)

    # ******************************* Contornos por visión artificial *******************************
    def procesar_contornos(self, frame):
        """Aplica preprocesamiento y dibuja las cajas delimitadoras de los dígitos."""
        img_dibujada = frame.copy()
        
        # 1. Escala de grises y desenfoque
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Binarización adaptativa
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. Dilatación ligera para conectar trazos
        kernel = np.ones((2, 2), np.uint8)
        processed_img = cv2.dilate(thresh, kernel, iterations=1)
        
        # 4. Encontrar contornos
        contours, _ = cv2.findContours(
            processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 5. Filtrar y dibujar según las heurísticas previas
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h)
            
            if area < 30:
                continue
                
            # Líneas horizontales (Rojo)
            if aspect_ratio > 3.0 and w > 40:
                cv2.rectangle(img_dibujada, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
            # Dígitos y Operadores (Verde)
            elif 0.2 < aspect_ratio < 1.8 and h > 15:
                cv2.rectangle(img_dibujada, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el resultado procesado en el panel derecho
        self.mostrar_en_label(img_dibujada, self.lbl_procesado)

    # Cambio de formato para QLabel
    def mostrar_en_label(self, img_cv, label):
        """Convierte OpenCV BGR a QPixmap y ajusta al QLabel."""
        rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio
        )
        label.setPixmap(pixmap)

    def closeEvent(self, e):
        if self.cap is not None:
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = AplicacionDeteccion()
    ventana.show()
    sys.exit(app.exec())
    print("Se aconseja para documentar en el código descargar la extensión \"Better coments\"")