# dataset_generator.py
import os
import json
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIGURACIÓN ==========
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

TEMAS = [
    "Inteligencia Artificial y ética",
    "Recetas de cocina italiana",
    "Historia del Imperio Romano",
    "Física cuántica para principiantes",
    "Estrategias de inversión financiera",
    "Entrenamiento de maratón",
    "Arquitectura moderna en ciudades",
    "Evolución de los videojuegos"
]

FRAGMENTOS_POR_TEMA = 5
PALABRAS_POR_FRAGMENTO = 60
INSTANCIAS = 10  # Número de secuencias a generar
OUTPUT_FILE = "dataset_segmentacion.json"

# ========== FUNCIÓN: GENERAR FRAGMENTOS ==========
def generar_fragmentos(tema, n=5):
    """Usa Groq para generar n fragmentos cortos sobre un mismo tema."""
    prompt = f"""
Genera {n} párrafos cortos y coherentes sobre el tema: "{tema}".
Cada párrafo debe tener entre {PALABRAS_POR_FRAGMENTO-10} y {PALABRAS_POR_FRAGMENTO+10} palabras.
Los párrafos deben estar relacionados temáticamente pero ser independientes (distintos subtemas o ángulos dentro del tema principal).

Debes responder ÚNICAMENTE con un formato JSON válido que contenga un objeto con la propiedad "fragmentos", cuyo valor sea una lista de strings. Ejemplo:
{{
  "fragmentos": [
    "Primer párrafo aquí...",
    "Segundo párrafo aquí..."
  ]
}}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    
    texto = response.choices[0].message.content.strip()
    datos = json.loads(texto)
    fragmentos = datos.get("fragmentos", [])
    return [f.strip() for f in fragmentos if f.strip()]

# ========== FUNCIÓN: CREAR SECUENCIA MEZCLADA ==========
def crear_secuencia():
    """
    Crea una secuencia mezclando fragmentos de 2-4 temas distintos.
    Retorna: (elementos[], cortes_ground_truth[])
    """
    num_temas = random.randint(2, 4)
    temas_elegidos = random.sample(TEMAS, num_temas)
    
    # Recolectar fragmentos de cada tema
    bloques = []
    for tema in temas_elegidos:
        # Tomamos 2-4 fragmentos consecutivos de cada tema
        n_frags = random.randint(2, 4)
        frags_disponibles = random.sample(banco_fragmentos[tema], min(n_frags, len(banco_fragmentos[tema])))
        bloques.append({
            "tema": tema,
            "fragmentos": frags_disponibles
        })
    
    # Mezclar bloques (cada bloque es un grupo coherente)
    random.shuffle(bloques)
    
    # Construir secuencia y ground truth
    elementos = []
    cortes = []  # Índices DONDE termina cada segmento (último índice del bloque)
    
    idx = 0
    for bloque in bloques:
        for frag in bloque["fragmentos"]:
            elementos.append({
                "id": idx,
                "texto": frag,
                "tema_real": bloque["tema"]
            })
            idx += 1
        cortes.append(idx - 1)  # El último índice de este bloque es un corte
    
    # El último índice no es un "corte" hacia adelante, así que lo removemos de cortes
    cortes_optimos = cortes[:-1]  # Cortes entre segmentos
    
    return {
        "elementos": elementos,
        "num_elementos": len(elementos),
        "temas_presentes": [b["tema"] for b in bloques],
        "cortes_ground_truth": cortes_optimos,  # Índices donde debe cortar
        "segmentos_ground_truth": [
            {"inicio": 0, "fin": cortes_optimos[0]} if cortes_optimos else {"inicio": 0, "fin": len(elementos)-1}
        ] + [
            {"inicio": cortes_optimos[i]+1, "fin": cortes_optimos[i+1]} 
            for i in range(len(cortes_optimos)-1)
        ] + ([
            {"inicio": cortes_optimos[-1]+1, "fin": len(elementos)-1}
        ] if cortes_optimos else [])
    }

# ========== EJECUCIÓN ==========
if __name__ == "__main__":
    print("🔄 Generando fragmentos temáticos con Groq...")
    
    # Generar banco de fragmentos por tema
    banco_fragmentos = {}
    for tema in TEMAS:
        print(f"  → Generando para: {tema}")
        try:
            frags = generar_fragmentos(tema, FRAGMENTOS_POR_TEMA)
            banco_fragmentos[tema] = frags
            print(f"     ✓ {len(frags)} fragmentos generados")
        except Exception as e:
            print(f"     ✗ Error: {e}")
            banco_fragmentos[tema] = []
    
    print(f"\n🔄 Creando {INSTANCIAS} secuencias de prueba...")
    dataset = []
    for i in range(INSTANCIAS):
        instancia = crear_secuencia()
        instancia["id"] = f"inst_{i:03d}"
        dataset.append(instancia)
        print(f"  → {instancia['id']}: {instancia['num_elementos']} elementos, "
              f"cortes en {instancia['cortes_ground_truth']}")
    
    # Guardar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset guardado en: {OUTPUT_FILE}")
    print(f"   Total instancias: {len(dataset)}")
    print(f"   Total elementos: {sum(d['num_elementos'] for d in dataset)}")
    
    # Mostrar ejemplo
    print("\n📋 Ejemplo de instancia:")
    ej = dataset[0]
    print(f"   ID: {ej['id']}")
    print(f"   Temas: {ej['temas_presentes']}")
    print(f"   Cortes esperados: {ej['cortes_ground_truth']}")
    for el in ej['elementos']:
        corte = " <-- CORTE" if el['id'] in ej['cortes_ground_truth'] else ""
        print(f"   [{el['id']}] ({el['tema_real'][:20]}...) {el['texto'][:60]}...{corte}")