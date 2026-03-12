import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class HASYv2Loader:
    def __init__(self, dataset_dir):
        """
        Inicializa el cargador apuntando al directorio raíz de HASYv2.
        El directorio debe contener la carpeta 'hasy-data' y el archivo 'hasy-data-labels.csv'.
        """
        self.dataset_dir = dataset_dir
        self.csv_path = os.path.join(dataset_dir, 'hasy-data-labels.csv')
        
        # Símbolos matemáticos en HASYv2 utilizados en el proyecto
        self.target_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\times']
        
    def load_and_preprocess(self, test_size=0.2, random_state=42):
        """
        Carga las imágenes, las filtra, normaliza y aplana en tensores 1D.
        """
        print("Cargando etiquetas del dataset HASYv2...")
        df = pd.read_csv(self.csv_path)
        
        # Filtrar el dataframe para conservar solo nuestras 11 clases objetivo
        df_filtered = df[df['latex'].isin(self.target_symbols)].copy()
        
        X = []
        y = []
        
        print(f"Procesando {len(df_filtered)} imágenes útiles...")
        
        for index, row in df_filtered.iterrows():
            img_path = os.path.join(self.dataset_dir, row['path'])
            
            # 1. Leer imagen en escala de grises
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # 2. Normalización Min-Max: Escalar píxeles al rango [0, 1]
            # Esto es vital para la estabilidad de los gradientes en el MLP
            img_normalized = img.astype(np.float32) / 255.0
            
            # 3. Flattening: Convertir matriz (32, 32) a un tensor 1D de (1024,)
            img_flattened = img_normalized.flatten()
            
            X.append(img_flattened)
            y.append(row['latex'])
            
        # Convertir listas a tensores de NumPy
        X = np.array(X)
        y = np.array(y)
        
        # 4. Codificar las etiquetas de texto ('0', '\times') a clases numéricas enteras
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 5. División del dataset (Train/Test Split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print("\n--- Resumen de Tensores Generados ---")
        print(f"Forma de X_train (Entrada): {X_train.shape}")
        print(f"Forma de y_train (Etiquetas): {y_train.shape}")
        print(f"Clases detectadas: {label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test, label_encoder

# Ejemplo de ejecución
if __name__ == "__main__":
    # Suponiendo que descargaste y extrajiste HASYv2 en la carpeta 'dataset/hasyv2'
    loader = HASYv2Loader(dataset_dir='archive')
    
    # Se genera la carga y preprocesamiento
    X_train, X_test, y_train, y_test, le = loader.load_and_preprocess()