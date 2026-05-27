import matplotlib.pyplot as plt
import pickle

# Cargar el dataset procesado
with open("dataset_con_embeddings.pkl", "rb") as f:
    dataset = pickle.load(f)

inst = dataset[0]
sim = inst["similitud_adyacente"]
cortes = inst["cortes_ground_truth"]

plt.figure(figsize=(10, 4))
plt.plot(range(len(sim)), sim, 'b-o', label='Similitud adyacente')
for c in cortes:
    plt.axvline(x=c, color='r', linestyle='--', alpha=0.7, label='Corte esperado' if c == cortes[0] else "")
plt.xlabel("Posición entre elementos")
plt.ylabel("Similitud coseno")
plt.title("Similitud adyacente vs Cortes ground truth")
plt.legend()
plt.savefig("similitud_adyacente.png")
print("✅ Gráfico guardado como 'similitud_adyacente.png'. Puedes abrir esta imagen para ver el resultado.")