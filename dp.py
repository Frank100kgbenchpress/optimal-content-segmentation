
"""
Tarea 3 (CORREGIDA): Segmentación Óptima con Programación Dinámica
Proyecto: Segmentación Óptima de Contenido

CORRECCIÓN: Coherencia de segmentos de 1 elemento = 0.0 (no hay pares internos).
Esto evita que el algoritmo prefiera trivialmente segmentos individuales.
"""

import pickle
import numpy as np
from collections import defaultdict
import json

# ========== CONFIGURACIÓN ==========
DATASET_INPUT = "dataset_con_embeddings.pkl"
RESULTADOS_OUTPUT = "resultados_segmentacion_fixed.json"

# Rango de λ ajustado: ahora la escala es diferente porque coherencia(1 elem)=0
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# ========== 1. CARGAR DATASET ==========
print("📂 Cargando dataset con embeddings...")
with open(DATASET_INPUT, "rb") as f:
    dataset = pickle.load(f)
print(f"   Instancias cargadas: {len(dataset)}\n")

# ========== 2. CORREGIR MATRIZ DE COHERENCIA ==========
print("🔧 Corrigiendo coherencia de segmentos unitarios a 0.0...")
for inst in dataset:
    coh = inst["coherencia_embedding"]
    n = coh.shape[0]
    for i in range(n):
        coh[i][i] = 0.0  # Un solo elemento no tiene coherencia interna
print("   Matrices corregidas.\n")

# ========== 3. ALGORITMO DP ==========
def segmentar_dp(elementos, coherencia_matrix, lam):
    """
    Programación Dinámica para segmentación óptima.
    
    C[j] = max_{0<=i<=j} ( C[i-1] + coherencia(i,j) - lam )
    
    Retorna: (valor_optimo, lista_de_cortes, lista_de_segmentos)
    """
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
    
    # Reconstrucción
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

# ========== 4. MÉTRICAS ==========
def boundary_f1(pred_cortes, gt_cortes, n, tolerancia=1):
    """Precision, recall y F1 para detección de fronteras (±tolerancia)."""
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
    """Pureza ponderada por tamaño de segmento."""
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

# ========== 5. EJECUTAR EXPERIMENTO ==========
print("🔬 Ejecutando experimentos con diferentes λ...\n")

resultados_por_lambda = {}

for lam in LAMBDAS:
    print(f"   λ = {lam:.1f}")
    stats = {
        "lambda": lam,
        "instancias": [],
        "f1_mean": 0.0,
        "precision_mean": 0.0,
        "recall_mean": 0.0,
        "purity_mean": 0.0,
        "seg_error_mean": 0.0
    }
    
    f1s, precs, recs, purities, seg_errors = [], [], [], [], []
    
    for inst in dataset:
        n = inst["num_elementos"]
        coh_matrix = inst["coherencia_embedding"]
        elementos = inst["elementos"]
        gt_cortes = inst["cortes_ground_truth"]
        
        valor_opt, pred_cortes, pred_segmentos = segmentar_dp(elementos, coh_matrix, lam)
        
        p, r, f1 = boundary_f1(pred_cortes, gt_cortes, n, tolerancia=1)
        pur = segment_purity(pred_segmentos, elementos)
        seg_err = (len(pred_cortes) + 1) - (len(gt_cortes) + 1)
        
        f1s.append(f1)
        precs.append(p)
        recs.append(r)
        purities.append(pur)
        seg_errors.append(seg_err)
        
        inst_result = {
            "id": inst["id"],
            "n_elementos": n,
            "gt_cortes": gt_cortes,
            "pred_cortes": pred_cortes,
            "n_segmentos_pred": len(pred_cortes) + 1,
            "n_segmentos_gt": len(gt_cortes) + 1,
            "f1": round(f1, 3),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "purity": round(pur, 3),
            "seg_error": seg_err,
            "valor_objetivo": round(valor_opt, 3)
        }
        stats["instancias"].append(inst_result)
    
    stats["f1_mean"] = round(np.mean(f1s), 3)
    stats["precision_mean"] = round(np.mean(precs), 3)
    stats["recall_mean"] = round(np.mean(recs), 3)
    stats["purity_mean"] = round(np.mean(purities), 3)
    stats["seg_error_mean"] = round(np.mean(seg_errors), 2)
    
    resultados_por_lambda[str(lam)] = stats
    print(f"      F1={stats['f1_mean']:.3f} | Prec={stats['precision_mean']:.3f} | Rec={stats['recall_mean']:.3f} | "
          f"Purity={stats['purity_mean']:.3f} | ΔSeg={stats['seg_error_mean']:+.1f}")

# ========== 6. RESUMEN COMPARATIVO ==========
print("\n" + "="*70)
print("📊 RESUMEN COMPARATIVO POR λ (Baseline - Solo Embeddings)")
print("="*70)
print(f"{'λ':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | {'Purity':>7} | {'ΔSeg':>6}")
print("-"*70)
for lam in LAMBDAS:
    s = resultados_por_lambda[str(lam)]
    print(f"{lam:>6.1f} | {s['f1_mean']:>6.3f} | {s['precision_mean']:>6.3f} | "
          f"{s['recall_mean']:>6.3f} | {s['purity_mean']:>7.3f} | {s['seg_error_mean']:>+6.1f}")

# ========== 7. DETALLE DE UN EJEMPLO ==========
best_lam = max(LAMBDAS, key=lambda l: resultados_por_lambda[str(l)]["f1_mean"])
print(f"\n🏆 Mejor λ según F1: {best_lam}")
print("\n📋 Detalle de instancia 0 con mejor λ:")
ej = resultados_por_lambda[str(best_lam)]["instancias"][0]
print(f"   GT cortes:      {ej['gt_cortes']}")
print(f"   Pred cortes:    {ej['pred_cortes']}")
print(f"   F1: {ej['f1']:.3f} | Purity: {ej['purity']:.3f} | ΔSeg: {ej['seg_error']:+d}")

# ========== 8. GUARDAR ==========
output = {
    "configuracion": "Baseline (solo embeddings)",
    "mejor_lambda_f1": float(best_lam),
    "lambdas_testeados": LAMBDAS,
    "nota": "Coherencia de segmentos unitarios corregida a 0.0",
    "resultados": resultados_por_lambda
}

with open(RESULTADOS_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n💾 Resultados guardados en: {RESULTADOS_OUTPUT}")
print("🎉 Tarea 3 corregida completada.")

