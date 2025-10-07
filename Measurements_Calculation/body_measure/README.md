@'
# Body Measure — README

Denne README er en **hurtig start** til at køre måle-pipelinen (front + side), med DeepLabV3-segmentering som standard, skala-fusion (ArUco + højde), debug-overlays og setup-profiler.

---

## Krav

- Python **3.9+**
- Pakker:
  - `numpy`, `opencv-contrib-python`
  - **Anbefalet:** `torch`, `torchvision` (DeepLabV3).
    - CPU-build (sikkert valg):
      ```powershell
      pip install "torch torchvision --index-url https://download.pytorch.org/whl/cpu"
      ```
    - CUDA (hvis du har NVIDIA-GPU): vælg korrekt cu-version, fx `cu121`:
      ```powershell
      pip install "torch torchvision --index-url https://download.pytorch.org/whl/cu121"
      ```

---

## Installation

Fra **projektroden** (Github\P7_UTRY_Py\Measurements_Calculation):
Dette vil sige at man skal åbne sin terminal projektroden for at køre det optimalt.

```powershell
pip install -e .

pip install numpy opencv-contrib-python

```
## Fil Oversigt
En hurtig oversigt over mapper og filers opsætning.
```powershell
project-root/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ body_measure/
│     ├─ cli.py
│     ├─ measure.py
│     ├─ segmenter.py
│     ├─ pose.py
│     ├─ geometry.py
│     └─ calibrate.py
└─ data/                # dine billeder
```

## Run with these commands

Alle eksempler er PowerShell-kommandoer kørt fra projektroden.
Sæt dine egne stier til billeder og profiler.

Anbefalet (DeepLabV3 + ArUco + Højde, gem profil, gem masker/overlays)
```powershell python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --aruco-mm 80 --height-cm 190 --backend deeplabv3 --device cpu --debug-dir ".\debug" --save-masks --setup-save ".\profiles\studio_A.json"

```
Forklaring: DeepLabV3 segmenterer. Skala = median af ArUco (80 mm) og højde (190 cm). Gemmer en setup-profil (ppc_ref). Skriver masker og overlays i .\debug\.

Kør med gemt setup-profil (hvis ArUco/Højde ikke oplyses)
```powershell
python -m body_measure.cli --front ".\data\front2.jpg" --side ".\data\side2.jpg" --setup-load ".\profiles\studio_A.json" --backend deeplabv3 --device cpu --debug-dir ".\debug"
```

Forklaring: Loader studio_A.json som ekstra skalakilde. Skriver overlays.

Kun ArUco (absolut skala)
```powershell 
python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --aruco-mm 100 --backend deeplabv3 --device cpu --debug-dir ".\debug" --save-masks
```

Forklaring: Bruger 100 mm (10 cm) ArUco-marker på kropsplan for stabil, absolut skala.

Kun Højde (hvis ingen marker)
```powershell
python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --height-cm 190 --backend deeplabv3 --device cpu --debug-dir ".\debug"
```

Forklaring: Skala fra personens rapporterede højde. Robust hvis afstand/pose er ens i front/side.

GPU (CUDA)
```powershell
python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --aruco-mm 80 --height-cm 190 --backend deeplabv3 --device cuda --debug-dir ".\debug"
```

Forklaring: Samme som anbefalet, men kører DeepLabV3 på GPU (kræver korrekt PyTorch CUDA-build).

Fallback uden PyTorch (tving OpenCV)
```powershell
python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --height-cm 190 --backend opencv --debug-dir ".\debug"
```

Forklaring: OpenCV HOG+GrabCut. Mindre præcis, men uden torch-afhængighed.

Fusion + profil-opdatering (læser og overskriver profil)
```powershell
python -m body_measure.cli --front ".\data\front.jpg" --side ".\data\side.jpg" --aruco-mm 80 --height-cm 190 --backend deeplabv3 --device cpu --debug-dir ".\debug" --save-masks --setup-load ".\profiles\studio_A.json" --setup-save ".\profiles\studio_A.json"
```

Forklaring: Bruger ArUco + højde + eksisterende profil som kilder, og overskriver profilen med ny samlet skala (median).
```
```
## Flag-reference (CLI)

- `--front PATH`, `--side PATH` — Front- og sidebillede (JPG/PNG).
- `--aruco-mm INT` — ArUco-markørens sidelængde i **millimeter** (fx 80 = 8 cm).
- `--height-cm FLOAT` — Personens højde i cm (fallback/ekstra skala).
- `--backend {deeplabv3|opencv|auto}` — Segmenteringsbackend (default: `deeplabv3`).
- `--device {cpu|cuda}` — Torch device til DeepLabV3 (default: `cpu`).
- `--debug-dir PATH` — Skriver `front_overlay.png` og `side_overlay.png`.  
  Brug `--save-masks` for også at gemme rå maske-PNGs.
- `--setup-save PATH.json` — Gemmer en **setup-profil** med feltet `ppc_ref` (pixels-per-cm).
- `--setup-load PATH.json` — Indlæser en setup-profil som **ekstra** skalakilde (fusion via median).

**Skala** beregnes som **median** af tilgængelige kilder (ArUco-front, ArUco-side, højde-front, højde-side, profil). Det gør løsningen robust over for mindre variationer i kameraafstand, zoom og højde.

---
```
```
## Debug-output

Når `--debug-dir` er sat, genereres:
- `front_overlay.png`, `side_overlay.png` — Silhuet (grøn), målelinjer, center-clip-vinduer og målte værdier.
- (valgfrit) `front_mask.png`, `side_mask.png` — Rå binære masker (`--save-masks`).

---
```
```
## Fejlfinding

- `No module named 'torch'` → installer `torch` og `torchvision` (se **Krav**).
- `DeepLabV3 kunne ikke initialiseres` → mismatch mellem torch/torchvision/CUDA; brug CPU-build eller match CUDA-version.
- `can't open/read file` → kontrollér stier (`Get-ChildItem`) eller brug absolutte stier i anførselstegn.
- Mål ser skæve ud → tjek overlays i `--debug-dir`; justér optagelse eller brug ArUco.

---
