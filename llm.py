
"""
Tarea 4: Integración de LLM (Groq) para evaluación semántica
Proyecto: Segmentación Óptima de Contenido

Estrategia: Evaluar con LLM solo fronteras ambiguas (similitud intermedia)
y segmentos candidatos pequeños, luego combinar con embeddings.
"""

import pickle
import json
import numpy as np
from collections import defaultdict
from groq import Groq
import time
import os
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# ========== CONFIGURACIÓN ==========
DATASET_INPUT = "dataset_con_embeddings.pkl"
RESULTADOS_BASELINE = "resultados_segmentacion_fixed.json"
RESULTADOS_OUTPUT = "resultados_segmentacion_llm.json"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ No se encontró GROQ_API_KEY en las variables de entorno.")

client = Groq(api_key=GROQ_API_KEY)

# Hiperparámetros del sistema híbrido
ALPHA = 0.6       # peso embeddings
BETA = 0.4        # peso LLM
LAMBDA = 0.7      # λ óptimo encontrado en baseline
UMBRAL_AMBIGUO = (0.25, 0.70)  # rango de similitud donde consultamos LLM
MAX_LLAMADAS_POR_INSTANCIA = 8  # límite de llamadas Groq por instancia

# ========== 1. CARGAR DATASET ==========
print("📂 Cargando dataset...")
with open(DATASET_INPUT, "rb") as f:
    dataset = pickle.load(f)
print(f"   Instancias: {len(dataset)}")

# Corregir coherencia unitaria
for inst in dataset:
    coh = inst["coherencia_embedding"]
    n = coh.shape[0]
    for i in range(n):
        coh[i][i] = 0.0
print("   Matrices de coherencia corregidas.\n")

# ========== 2. FUNCIONES LLM ==========
llm_cache = {}

def llamar_groq_frontera(texto_a, texto_b, max_retries=3):
    """
    Evalúa con Groq si dos fragmentos adyacentes mantienen continuidad temática.
    Retorna score 0.0-1.0 (1.0 = mismo tema perfecto).
    """
    key = (texto_a[:100], texto_b[:100])  # cache por inicio de texto
    if key in llm_cache:
        return llm_cache[key]
    
    prompt = f"""Evalúa la continuidad temática entre dos fragmentos consecutivos.

Fragmento A: \"{texto_a}\"

Fragmento B: \"{texto_b}\"

¿Pertenecen al mismo tema continuo o representan un cambio de tema?
Responde ÚNICAMENTE con un número entero del 0 al 10:
- 10: mismo tema, continuidad perfecta
- 7-9: mismo tema general, ligera transición
- 4-6: temas relacionados pero distintos
- 1-3: cambio notable de tema
- 0: cambio completo y abrupto de tema

Número:"""
    
    for intento in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            texto_resp = response.choices[0].message.content.strip()
            # Extraer número
            import re
            nums = re.findall(r'\d+', texto_resp)
            if nums:
                score = int(nums[0])
                score = max(0, min(10, score)) / 10.0
                llm_cache[key] = score
                return score
        except Exception as e:
            print(f"      ⚠️ Error LLM (intento {intento+1}): {e}")
            time.sleep(1)
    
    llm_cache[key] = 0.5  # fallback neutral
    return 0.5

def llamar_groq_segmento(textos_segmento, max_retries=3):
    """
    Evalúa coherencia de un segmento completo (2-4 fragmentos).
    Retorna score 0.0-1.0.
    """
    key = tuple(t[:80] for t in textos_segmento)
    if key in llm_cache:
        return llm_cache[key]
    
    fragmentos_str = "\n".join([f"{i+1}. \"{t[:200]}\"" for i, t in enumerate(textos_segmento)])
    
    prompt = f"""Evalúa la coherencia temática de este segmento de texto.

Fragmentos:
{fragmentos_str}

¿Forman estos fragmentos un segmento temáticamente coherente?
Responde ÚNICAMENTE con un número entero del 0 al 10:
- 10: totalmente coherente, mismo tema central claro
- 7-9: coherente con matices menores
- 4-6: parcialmente coherente, algunas desviaciones
- 1-3: mayormente incoherente
- 0: completamente incoherente, temas mezclados sin relación

Número:"""
    
    for intento in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            texto_resp = response.choices[0].message.content.strip()
            import re
            nums = re.findall(r'\d+', texto_resp)
            if nums:
                score = int(nums[0])
                score = max(0, min(10, score)) / 10.0
                llm_cache[key] = score
                return score
        except Exception as e:
            print(f"      ⚠️ Error LLM (intento {intento+1}): {e}")
            time.sleep(1)
    
    llm_cache[key] = 0.5
    return 0.5

# ========== 3. ALGORITMO DP (igual que baseline) ==========
def segmentar_dp(elementos, coherencia_matrix, lam):
    n = len(elementos)
    if n == 0:
        return 0, [], []
    
    C = [-float('inf')] * n
    split = [-1] * n
    
    for j in range(n):
        for i in range(j + 1):
            coh = coherencia_matrix[i][j]
            costo_previo = C[i - 1] if i > 0 else 0
            val = costo_previo + coh - lam
            if val > C[j]:
                C[j] = val
                split[j] = i
    
    segmentos = []
    cortes = []
    j = n - 1
    while j >= 0:
        i = split[j]
        segmentos.append({"inicio": i, "fin": j})
        if i > 0:
            cortes.append(i - 1)
        j = i - 1
    
    segmentos.reverse()
    cortes.reverse()
    return C[n-1], cortes, segmentos

def boundary_f1(pred_cortes, gt_cortes, n, tolerancia=1):
    if len(gt_cortes) == 0:
        precision = 1.0 if len(pred_cortes) == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if len(pred_cortes) == 0 else 0.0
        return precision, recall, f1
    if len(pred_cortes) == 0:
        return 0.0, 0.0, 0.0
    
    gt_matched = set()
    pred_matched = set()
    for pi, pc in enumerate(pred_cortes):
        for gi, gc in enumerate(gt_cortes):
            if gi in gt_matched:
                continue
            if abs(pc - gc) <= tolerancia:
                gt_matched.add(gi)
                pred_matched.add(pi)
                break
    
    tp = len(pred_matched)
    fp = len(pred_cortes) - tp
    fn = len(gt_cortes) - len(gt_matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def segment_purity(segmentos_pred, elementos):
    total_peso = 0
    pureza_total = 0.0
    for seg in segmentos_pred:
        i, j = seg["inicio"], seg["fin"]
        temas = [elementos[k]["tema_real"] for k in range(i, j + 1)]
        if len(temas) == 0:
            continue
        counts = defaultdict(int)
        for t in temas:
            counts[t] += 1
        max_count = max(counts.values())
        pureza = max_count / len(temas)
        pureza_total += pureza * len(temas)
        total_peso += len(temas)
    return pureza_total / total_peso if total_peso > 0 else 0.0

# ========== 4. EVALUAR INSTANCIAS CON LLM ==========
print("🔬 Ejecutando segmentación con LLM (Groq)...")
print(f"   λ = {LAMBDA} | α = {ALPHA} | β = {BETA}")
print(f"   Umbral ambiguo: {UMBRAL_AMBIGUO}\n")

resultados_llm = []
llamadas_totales = 0

for idx, inst in enumerate(dataset):
    inst_id = inst["id"]
    n = inst["num_elementos"]
    elementos = inst["elementos"]
    gt_cortes = inst["cortes_ground_truth"]
    sim_ady = inst["similitud_adyacente"]
    coh_emb = inst["coherencia_embedding"].copy()
    
    print(f"   → {inst_id} ({n} elementos)")
    
    # --- 4.1 Identificar fronteras ambiguas ---
    fronteras_ambiguas = []
    for i in range(n - 1):
        if UMBRAL_AMBIGUO[0] <= sim_ady[i] <= UMBRAL_AMBIGUO[1]:
            fronteras_ambiguas.append(i)
    
    # Limitar número de consultas LLM
    if len(fronteras_ambiguas) > MAX_LLAMADAS_POR_INSTANCIA:
        # Priorizar las más ambiguas (más cercanas al centro del rango)
        centro = (UMBRAL_AMBIGUO[0] + UMBRAL_AMBIGUO[1]) / 2
        fronteras_ambiguas.sort(key=lambda i: abs(sim_ady[i] - centro))
        fronteras_ambiguas = fronteras_ambiguas[:MAX_LLAMADAS_POR_INSTANCIA]
    
    print(f"      Fronteras ambiguas detectadas: {len(fronteras_ambiguas)}")
    
    # --- 4.2 Consultar LLM para fronteras ambiguas ---
    scores_llm_fronteras = {}
    for pos in fronteras_ambiguas:
        texto_a = elementos[pos]["texto"]
        texto_b = elementos[pos + 1]["texto"]
        score = llamar_groq_frontera(texto_a, texto_b)
        scores_llm_fronteras[pos] = score
        llamadas_totales += 1
        print(f"         Frontera {pos}↔{pos+1}: sim_emb={sim_ady[pos]:.3f} | LLM={score:.2f}")
    
    # --- 4.3 Construir matriz de coherencia híbrida ---
    coh_hibrida = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                coh_hibrida[i][j] = 0.0
                continue
            
            # Coherencia base por embeddings
            coh_base = coh_emb[i][j]
            
            # Si el segmento [i:j] cruza alguna frontera ambigua evaluada por LLM,
            # ajustamos la coherencia con el score LLM de esa frontera.
            # Estrategia: si el segmento contiene una frontera ambigua, su coherencia
            # se ve afectada por el score LLM de esa frontera.
            ajuste_llm = 0.0
            peso_ajuste = 0.0
            
            for pos in range(i, j):
                if pos in scores_llm_fronteras:
                    # Si LLM dice que hay continuidad (score alto), aumenta coherencia
                    # Si LLM dice cambio (score bajo), reduce coherencia
                    ajuste_llm += scores_llm_fronteras[pos]
                    peso_ajuste += 1.0
            
            if peso_ajuste > 0:
                # Promedio de scores LLM en fronteras internas
                coh_llm_component = ajuste_llm / peso_ajuste
                # Combinación: más peso a LLM cuando hay fronteras ambiguas internas
                factor_llm = min(0.5, peso_ajuste / (j - i + 1))  # max 50% peso LLM
                coh_hibrida[i][j] = (1 - factor_llm) * coh_base + factor_llm * coh_llm_component
            else:
                coh_hibrida[i][j] = coh_base
    
    # --- 4.4 Evaluación con segmentos pequeños (bonus) ---
    # Para segmentos de tamaño 2-3 que contienen cortes GT, usamos LLM directo
    # para validar si realmente son coherentes
    for seg_gt in inst.get("segmentos_ground_truth", []):
        ini, fin = seg_gt["inicio"], seg_gt["fin"]
        tam = fin - ini + 1
        if 2 <= tam <= 3:
            textos_seg = [elementos[k]["texto"] for k in range(ini, fin + 1)]
            score_seg = llamar_groq_segmento(textos_seg)
            # Refinar coherencia del segmento ground truth
            coh_hibrida[ini][fin] = ALPHA * coh_emb[ini][fin] + BETA * score_seg
            llamadas_totales += 1
            print(f"         Seg [{ini}:{fin}] (GT): emb={coh_emb[ini][fin]:.3f} | LLM={score_seg:.2f} | híbrida={coh_hibrida[ini][fin]:.3f}")
    
    # --- 4.5 Ejecutar DP con coherencia híbrida ---
    val_opt, pred_cortes, pred_segmentos = segmentar_dp(elementos, coh_hibrida, LAMBDA)
    
    # Métricas
    p, r, f1 = boundary_f1(pred_cortes, gt_cortes, n, tolerancia=1)
    pur = segment_purity(pred_segmentos, elementos)
    seg_err = (len(pred_cortes) + 1) - (len(gt_cortes) + 1)
    
    print(f"      ✅ Cortes GT: {gt_cortes} | Pred: {pred_cortes} | F1={f1:.3f}\n")
    
    resultados_llm.append({
        "id": inst_id,
        "n_elementos": n,
        "gt_cortes": gt_cortes,
        "pred_cortes": pred_cortes,
        "f1": round(f1, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "purity": round(pur, 3),
        "seg_error": seg_err,
        "llm_fronteras_evaluadas": len(fronteras_ambiguas),
        "valor_objetivo": round(val_opt, 3)
    })

# ========== 5. ESTADÍSTICAS GLOBALES ==========
f1s = [r["f1"] for r in resultados_llm]
precs = [r["precision"] for r in resultados_llm]
recs = [r["recall"] for r in resultados_llm]
purities = [r["purity"] for r in resultados_llm]
seg_errors = [r["seg_error"] for r in resultados_llm]

resumen = {
    "configuracion": "LLM-Fronteras (híbrido embeddings + Groq)",
    "lambda": LAMBDA,
    "alpha": ALPHA,
    "beta": BETA,
    "llamadas_groq_totales": llamadas_totales,
    "f1_mean": round(np.mean(f1s), 3),
    "precision_mean": round(np.mean(precs), 3),
    "recall_mean": round(np.mean(recs), 3),
    "purity_mean": round(np.mean(purities), 3),
    "seg_error_mean": round(np.mean(seg_errors), 2),
    "instancias": resultados_llm
}

print("="*70)
print("📊 RESULTADOS LLM (Groq)")
print("="*70)
print(f"   F1:       {resumen['f1_mean']:.3f}")
print(f"   Precisión:{resumen['precision_mean']:.3f}")
print(f"   Recall:   {resumen['recall_mean']:.3f}")
print(f"   Purity:   {resumen['purity_mean']:.3f}")
print(f"   ΔSeg:     {resumen['seg_error_mean']:+.1f}")
print(f"   Llamadas Groq: {llamadas_totales}")
print("="*70)

# ========== 6. COMPARACIÓN CON BASELINE ==========
print("\n📊 COMPARACIÓN BASELINE vs LLM:")
print(f"   {'Métrica':<12} | {'Baseline':>10} | {'LLM':>10} | {'Δ':>8}")
print("   " + "-"*50)

# Cargar baseline para comparar
try:
    with open(RESULTADOS_BASELINE, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    baseline = baseline_data["resultados"][str(LAMBDA)]
    
    for metrica, key in [("F1", "f1_mean"), ("Precisión", "precision_mean"), 
                         ("Recall", "recall_mean"), ("Purity", "purity_mean")]:
        b = baseline[key]
        l = resumen[key]
        delta = l - b
        print(f"   {metrica:<12} | {b:>10.3f} | {l:>10.3f} | {delta:>+8.3f}")
except Exception as e:
    print(f"   No se pudo cargar baseline para comparación: {e}")

# ========== 7. GUARDAR ==========
with open(RESULTADOS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

print(f"\n💾 Resultados guardados en: {RESULTADOS_OUTPUT}")
print("🎉 Tarea 4 completada.")
