"""
Tarea 2: Embeddings y Matriz de Coherencia Local
Proyecto: Segmentación Óptima de Contenido
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ========== CONFIGURACIÓN ==========
DATASET_INPUT = "dataset_segmentacion.json"
DATASET_OUTPUT = "dataset_con_embeddings.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # ~80MB, multilingüe básico, muy rápido "all-MiniLM-L6-v2"
# Alternativa mejor para español: "paraphrase-multilingual-MiniLM-L12-v2"

# ========== 1. CARGAR DATASET ==========
print("📂 Cargando dataset...")
with open(DATASET_INPUT, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"   Instancias cargadas: {len(dataset)}")

# ========== 2. CARGAR MODELO DE EMBEDDINGS ==========
print(f"🧠 Cargando modelo de embeddings: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("   Modelo listo.")

# ========== 3. PROCESAR CADA INSTANCIA ==========
for inst in dataset:
    inst_id = inst["id"]
    textos = [el["texto"] for el in inst["elementos"]]
    n = len(textos)
    
    # --- 3.1 Embeddings de cada elemento ---
    embeddings = model.encode(textos, convert_to_numpy=True, show_progress_bar=False)
    inst["embeddings"] = embeddings  # shape: (n, 384)
    
    # --- 3.2 Similitud entre ADYACENTES (local) ---
    # sim_adyacente[i] = similitud entre elemento i e i+1
    sim_adyacente = []
    for i in range(n - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i+1].reshape(1, -1)
        )[0, 0]
        sim_adyacente.append(float(sim))
    inst["similitud_adyacente"] = sim_adyacente  # length: n-1
    
    # --- 3.3 Similitud GLOBAL (todos vs todos) ---
    sim_global = cosine_similarity(embeddings)  # shape: (n, n)
    inst["similitud_global"] = sim_global
    
    # --- 3.4 Coherencia de segmentos candidatos (precálculo) ---
    # Para acelerar la Programación Dinámica, precomputamos la coherencia
    # de cualquier segmento [i:j] como el promedio de similitudes internas.
    # coherence[i][j] = coherencia del segmento desde i hasta j (inclusive)
    coherence_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                coherence_matrix[i][j] = 1.0  # un solo elemento es trivialmente coherente
            else:
                # Extraer submatriz de similitud global entre i y j
                sub = sim_global[i:j+1, i:j+1]
                # Promedio de similitudes estrictamente superiores (sin diagonal)
                mask = ~np.eye(sub.shape[0], dtype=bool)
                if np.any(mask):
                    coherence_matrix[i][j] = float(np.mean(sub[mask]))
                else:
                    coherence_matrix[i][j] = 1.0
    inst["coherencia_embedding"] = coherence_matrix
    
    # --- 3.5 Estadísticas para análisis ---
    inst["stats"] = {
        "num_elementos": n,
        "sim_adyacente_promedio": float(np.mean(sim_adyacente)) if sim_adyacente else 0.0,
        "sim_adyacente_min": float(np.min(sim_adyacente)) if sim_adyacente else 0.0,
        "sim_adyacente_max": float(np.max(sim_adyacente)) if sim_adyacente else 0.0,
    }
    
    print(f"   ✅ {inst_id}: {n} elementos, sim_ady promedio={inst['stats']['sim_adyacente_promedio']:.3f}")

# ========== 4. GUARDAR RESULTADO ==========
print(f"\n💾 Guardando dataset procesado en: {DATASET_OUTPUT}")
with open(DATASET_OUTPUT, "wb") as f:
    pickle.dump(dataset, f)

print("\n🎉 Tarea 2 completada.")
print(f"   Archivo generado: {DATASET_OUTPUT}")
print("   Contiene: embeddings, similitud local, similitud global, coherencia precomputada.")

# ========== 5. VISUALIZACIÓN DE EJEMPLO ==========
print("\n📊 Ejemplo de similitudes adyacentes (instancia 0):")
ej = dataset[0]
for i, sim in enumerate(ej["similitud_adyacente"]):
    corte_gt = " <-- CORTE ESPERADO" if i in ej["cortes_ground_truth"] else ""
    print(f"   Elemento {i} ↔ {i+1}: {sim:.3f}{corte_gt}")

print("\n📐 Ejemplo de coherencia por segmentos (instancia 0):")
coh = ej["coherencia_embedding"]
for i in range(min(4, len(ej["elementos"]))):
    for j in range(i, min(i+4, len(ej["elementos"]))):
        print(f"   Segmento [{i}:{j}] = {coh[i][j]:.3f}")

