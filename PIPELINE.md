# Pipeline — Resumo

```
┌─────────────────────────────────────────────────────────────────┐
│  vídeo .mp4                                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [0] CALIBRAÇÃO HOMOGRAFIA   (manual, 1x por vídeo)              │
│     run_homography.py                                           │
│     • select_points.py    → matplotlib, clica pontos video↔campo│
│     • compute_homography  → cv2.findHomography (RANSAC)         │
│     → output/homography.json                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] TRACKING                                                    │
│     src/pipeline_sam2.py                                        │
│     • YOLOv8l (treinado)  → deteta jogadores no melhor frame    │
│     • SAM2 (base+)        → propaga máscaras pelo vídeo todo    │
│     • Processa em chunks  → evita OOM em vídeos longos          │
│     → trajectories_<video>.json  (bboxes em pixels)             │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] HOMOGRAFIA → METROS                                         │
│     homography/transform.py                                     │
│     • SIFT por frame      → compensa pan da câmara              │
│     • Suavização gaussiana das H (filtro temporal)              │
│     • Kalman (vel. const.) + innovation gate → suaviza ruído    │
│     • Pés (centro-baixo da bbox) → ponto no chão                │
│     → trajectories_2d.json  (coords em metros, aproximadas)     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] VALIDAÇÃO  (interativo)                                     │
│     homography/validate.py                                      │
│     • matplotlib  → clica num jogador no frame                  │
│     • Gera vídeo lado-a-lado: original | campo 2D + posição     │
│     → validation_track<N>.mp4                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Comandos

```bash
# Calibração (1x por vídeo)
python run_homography.py videos/<video>.mp4

# Pipeline completo
python run.py

# Partes
python run.py --only 1        # só tracking
python run.py --from 2        # do passo 2 até ao fim
python run.py --only 3        # só validação

# Opcional (só quando a homografia está mal calibrada)
python homography/correct_coords.py
```

## Tecnologias

| Bloco | Lib                  | Função                              |
|-------|----------------------|-------------------------------------|
| Det.  | ultralytics (YOLOv8) | Deteta jogadores no frame inicial   |
| Track | Meta SAM2 (base+)    | Propaga máscaras (mantém identidade)|
| Hom.  | OpenCV               | findHomography + SIFT + perspectiveTransform |
| Suav. | scipy + cv2.Kalman   | gaussian_filter1d (H) + Kalman (xy) |
| Vis.  | matplotlib + cv2     | UI interativa + render vídeo        |

## Ficheiros importantes

```
videos/<video>.mp4                 → input
output/homography_points.json      → pontos manuais (calibração)
output/homography.json             → matriz H
output/trajectories_<video>.json   → tracking em pixels
output/trajectories_2d.json        → metros (final, com correções)
output/validation_track<N>.mp4     → vídeo de validação
```
