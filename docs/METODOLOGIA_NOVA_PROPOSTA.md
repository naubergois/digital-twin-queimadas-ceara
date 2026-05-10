# Nova metodologia proposta — deteção de queimadas com GOES-16 (Ceará)

## Porque as métricas continuam fracas

O pipeline anterior compara **uma grade binária derivada de uma ou poucas imagens térmicas** com **focos INPE agregados por dia**, em células ~0,05–0,1°. Isto introduz **quatro desalinhamentos estruturais**:

1. **Tempo** — O foco INPE tem hora; o CMIPF é um intervalo de ~10 min noutro instante. Sem cubo horário denso, há falsos negativos inevitáveis.  
2. **Semântica** — Anomalia térmica ≠ queimada (solo quente, urbano, borda de nuvem, reflexão).  
3. **Escala** — O foco é quase pontual; a célula é grande. Dilatar a verdade melhora números mas não resolve o desencontro físico.  
4. **Produto** — CMIPF é brilho/temperatura de cena; produtos **AF (Active Fire)** ou algoritmos NOAA dedicados filtram melhor que um limiar único.

Por isso F1 moderado (ex.: 0,08–0,15) **não implica** falha trivial do código — reflecte um problema **mal posto** se o objectivo é “copiar o INPE pixel a pixel”.

---

## Linha A — Manter (curto prazo)

- DQF, multi-banda 7/13/14, várias horas UTC, validação com dilatação opcional e relatórios com calibração explícita.  
- Útil para **exploração** e **comparação de métodos** no mesmo dia.

---

## Linha B — Nova metodologia operacional (recomendada): **PEAK + PERSIST + FUSÃO**

**Ideia:** Um foco activo costuma (não sempre) produzir:

1. **Pico** de anomalia térmica em pelo menos um granulo;  
2. **Persistência** — sinal anómalo em **várias horas** do mesmo dia (nuvem e ruído pontual tendem a ser mais erráticos);  
3. **Contraste espectral** (BT7−BT14) já embutido no score horário multi-escala.

### Passos (não supervisionados)

1. Por cada hora \(t\) e célula, calcular o score \(s_t\) já definido em `hourly_anomaly_score` (multi-escala em T_B13 + BTD ponderado).  
2. **Activar** hora \(t\) na célula se \(s_t \ge P_{q}(s_t \mid \text{válido})\) (ex.: \(q=82\)).  
3. **Persistência** \(p = \frac{1}{T}\sum_t \mathbf{1}[\text{activa em } t]\).  
4. **Fusão**  
   \[
   C = w_p \cdot \hat{S}_{\max} + w_m \cdot \hat{S}_{\text{média}} + w_r \cdot p
   \]
   com \(\hat{S}\) normalização robusta (percentis 4–96) na cena.  
5. **Limiar** global em \(C\) (percentil derivado de `contamination`).  
6. **Portão de persistência** — se \(T \ge 2\): exigir \(p \ge p_{\min}\) (ex.: \(\max(0{,}22,\,0{,}85/T)\)). Com uma só hora, o portão desliga-se.  
7. **Morfologia** — `binary_opening` 3×3 (remove “sal e pimenta” espacial).

Implementação: método **`combined_persistence`** (`src/goes_fire_method_v2.py`).

---

## Linha C — Médio prazo (melhor ciência)

| Acção | Benefício |
|-------|-----------|
| **Cubo GOES** com passo horário (vários dias) por célula | Baseline sazonal / horária; z-score sem INPE. |
| **Produto AF oficial** (quando disponível no mesmo bucket / catálogo) | Alinhamento semântico com “fogo”. |
| **Validação centrada em eventos** | Para cada foco INPE: acerto se qualquer célula num raio \(R\) km e janela \(\pm \Delta h\) tem score alto (métrica mais honesta). |
| **Modelo generativo leve** (PCA robusta / IF em traços temporais por célula) | Separa “modo normal” de “ruptura”. |

---

## Linha D — O que não resolver só com “mais ML”

- Sem **alinhamento temporal** credível ou produto AF, ganhos de F1 na grade diária terão tecto baixo.  
- **Calibração com INPE no mesmo dia** serve para relatório, não para generalização operacional.

---

## Como correr a Linha B

```bash
python -m src.unsupervised_fire_goes \
  --inpe-csv data/inpe_focos_ce/focos_ce_INPE_2024_2026.csv \
  --dates 2024-10-31 \
  --hours-utc 16,17,18 \
  --channels 7,13,14 \
  --method combined_persistence \
  --truth-dilate 1 \
  --output-json data/goes16_eval/metrics_combined_persistence.json \
  --map-dir data/goes16_eval/maps
```

Comparar com `--method all` (métodos anteriores + twin).

---

*Documento de pesquisa — evoluir com resultados em `EVOLUCAO_PESQUISA.md`.*
