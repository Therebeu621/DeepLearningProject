from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd
import numpy as np
import librosa
import torch
import os


# Assurez-vous que le chemin est correct après avoir décompressé le fichier
metadata_path = 'C:/Users/DELL/PycharmProjects/DeepLearningProject/audio/UrbanSound8K.csv'

# Charger le fichier CSV dans un DataFrame pandas
df = pd.read_csv(metadata_path)

# Afficher les 5 premières lignes pour voir la structure
print(df.head())

# Afficher les différentes classes de sons disponibles
print("\nClasses de sons disponibles :")
print(df['class'].unique())

# Chemin vers votre dossier audio
audio_folder = 'C:/Users/DELL/PycharmProjects/DeepLearningProject/audio/fold1/'
file_name = '7383-3-0-0.wav' # Un fichier d'aboiement de chien pour l'exemple
audio_path = audio_folder + file_name

# Charger le fichier audio
# sr=None permet de garder le taux d'échantillonnage d'origine
audio_signal, sampling_rate = librosa.load(audio_path, sr=None)

print("Signal chargé :", audio_signal)
print("Taux d'échantillonnage :", sampling_rate)

# Calculer le Mel-spectrogramme
mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sampling_rate)

# Convertir en décibels (dB) pour une meilleure visualisation
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Afficher le spectrogramme
librosa.display.specshow(mel_spectrogram_db, sr=sampling_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogramme')
plt.tight_layout()
plt.show()


class UrbanSound8KDataset(Dataset):
    def __init__(self, annotations_df, audio_dir, target_length=173):  # On ajoute target_length
        self.annotations = annotations_df
        self.audio_dir = audio_dir
        self.target_length = target_length  # On stocke la longueur cible

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # ... (les 4 premières étapes ne changent pas)
        file_name = self.annotations.iloc[index, 0]
        fold_number = f"fold{self.annotations.iloc[index, 5]}"
        audio_path = os.path.join(self.audio_dir, fold_number, file_name)

        signal, sr = librosa.load(audio_path, sr=None)

        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        label = self.annotations.iloc[index, 6]

        # --- NOUVELLE PARTIE : UNIFORMISATION DE LA TAILLE ---
        spectrogram_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32)

        # Si le spectrogramme est trop long, on le tronque
        if spectrogram_tensor.shape[1] > self.target_length:
            spectrogram_tensor = spectrogram_tensor[:, :self.target_length]

        # S'il est trop court, on ajoute du padding
        elif spectrogram_tensor.shape[1] < self.target_length:
            padding_needed = self.target_length - spectrogram_tensor.shape[1]
            # F.pad ajoute du padding (ici, à droite du spectrogramme)
            spectrogram_tensor = F.pad(spectrogram_tensor, (0, padding_needed))

        return spectrogram_tensor, torch.tensor(label)


# --- Comment l'utiliser ---
if __name__ == '__main__':
    CSV_PATH = 'C:/Users/DELL/PycharmProjects/DeepLearningProject/audio/UrbanSound8K.csv'
    AUDIO_DIR = 'C:/Users/DELL/PycharmProjects/DeepLearningProject/audio/'

    # Créer une instance de votre dataset
    urbansound_dataset = UrbanSound8KDataset(annotations_file_path=CSV_PATH, audio_dir=AUDIO_DIR)

    print(f"Il y a {len(urbansound_dataset)} échantillons dans le dataset.")

    # Récupérer le premier échantillon (spectrogramme + étiquette)
    spectrogram, label = urbansound_dataset[0]

    print("\nExemple du premier échantillon :")
    print("Shape du spectrogramme:", spectrogram.shape)
    print("Étiquette:", label)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # On peut rendre le nb de classes paramétrable
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Cette couche va redimensionner la sortie à 7x7
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            # La taille d'entrée est maintenant fixe et connue ! 32 canaux * 7 * 7
            nn.Linear(in_features=32 * 7 * 7, out_features=num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # On ajoute la couche adaptative ici
        logits = self.linear_layers(x)
        return logits

# Pour instancier le modèle :
model = SimpleCNN()
print(model)

from torch.utils.data import DataLoader, Subset

# Filtrer le DataFrame pandas pour créer les deux ensembles
train_df = df[df['fold'] != 10]
val_df = df[df['fold'] == 10]

# Créer des instances de Dataset pour chacun
# Note: On doit adapter la classe Dataset pour qu'elle accepte un DataFrame
# Voici une version modifiée de votre Dataset :
class UrbanSound8KDataset(Dataset):
    def __init__(self, annotations_df, audio_dir):
        self.annotations = annotations_df
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # ... (le reste de la méthode __getitem__ ne change pas)
        file_name = self.annotations.iloc[index, 0]
        fold_number = f"fold{self.annotations.iloc[index, 5]}"
        audio_path = os.path.join(self.audio_dir, fold_number, file_name)
        signal, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        label = self.annotations.iloc[index, 6]
        return torch.tensor(mel_spectrogram_db, dtype=torch.float32), torch.tensor(label)


# Créer les datasets
train_dataset = UrbanSound8KDataset(annotations_df=train_df, audio_dir=AUDIO_DIR)
val_dataset = UrbanSound8KDataset(annotations_df=val_df, audio_dir=AUDIO_DIR)

# Créer les DataLoaders
BATCH_SIZE = 64
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# S'assurer que le modèle tourne sur le bon appareil (GPU si disponible, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de l'appareil : {device}")

model = SimpleCNN().to(device)

# Définir la fonction de coût et l'optimiseur
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()  # Mettre le modèle en mode entraînement
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Envoyer les données sur le bon appareil

        # 1. Calculer la prédiction et la perte
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # 2. Rétropropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Perte moyenne d'entraînement: {avg_loss:.4f}")


def validation_loop(dataloader, model, loss_fn):
    model.eval()  # Mettre le modèle en mode évaluation
    total_loss, correct = 0, 0
    num_samples = 0
    with torch.no_grad():  # On ne calcule pas les gradients en validation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_samples += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / num_samples
    print(f"Précision: {(100 * accuracy):>0.1f}%, Perte moyenne de validation: {avg_loss:.4f}\n")


# Lancer l'entraînement pour plusieurs époques
epochs = 10
for t in range(epochs):
    print(f"--- Époque {t + 1} ---")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    validation_loop(val_dataloader, model, loss_fn)

print("Entraînement terminé !")

# Sauvegarder le modèle entraîné (très important pour le projet !)
torch.save(model.state_dict(), "modele_specialiste.pth")
print("Modèle sauvegardé sous modele_specialiste.pth")


train_dataset = UrbanSound8KDataset(annotations_df=train_df, audio_dir=AUDIO_DIR)
val_dataset = UrbanSound8KDataset(annotations_df=val_df, audio_dir=AUDIO_DIR)
