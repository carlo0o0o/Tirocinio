import numpy as np
import matplotlib.pyplot as plt

# Carica i dati salvati
data = np.load('results/20250515T141927Z.VUAH/results.npz')    #IA addestrata da plottare 

# Stampa le chiavi disponibili nel file .npz
print("Chiavi disponibili nel file .npz:", data.files)



epochs = range(1, len(data['f1']) + 1)

# Grafico della Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs, data['loss_train'], label='Train Loss')
plt.plot(epochs, data['loss_valid'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# Grafico dell'F1-score
plt.subplot(1, 3, 2)
plt.plot(epochs, data['f1'], label='F1-score', color='green')
plt.xlabel('Epochs')
plt.ylabel('F1-score')
plt.title('F1-score Trend')
plt.legend()

# Grafico della Precision & Recall
plt.subplot(1, 3, 3)
plt.plot(epochs, data['precision'], label='Precision', color='red')
plt.plot(epochs, data['recall'], label='Recall', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Precision & Recall')
plt.legend()

plt.tight_layout()
plt.savefig('results/training_plots.png')  # Salva il grafico come immagine sovrascrive lo stesso file 
plt.show()
