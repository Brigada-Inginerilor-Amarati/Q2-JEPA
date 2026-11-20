# Gate-ul GPU: timpi, precizie și ce decide pentru grid (20 noiembrie 2026)

Rezultatele primei rulări pe GPU a componentelor cuantice și a gate-ului de timing HQViT din
[phase-2 spec](2026-11-03-q2jepa-phase2-design.md) §6. Ce scrie aici completează spec-ul de phase 2, nu îl
înlocuiește.

**Platformă:** Google Colab Pro, Tesla T4 (15 GB), ~1,15 unități pe oră. Cod de pe `main` la `0ab636a`. Target
CUDA-Q `nvidia`; `cuda-quantum` 0.16, wheel-ul ales de `scripts/colab_setup.py`.

## 1. Corectitudine pe GPU

Testele cuantice rulate pe `nvidia`:

| Teste | Precizie | Rezultat |
|---|---|---|
| `test_born`, `test_attention_equivalence` | fp32 (implicit) | 14 passed |
| `test_qcbm`, `test_hqvit_kernels`, `test_hqvit` | fp32 (implicit) | 18 passed, **5 failed** |
| `test_qcbm`, `test_hqvit_kernels`, `test_hqvit` | **fp64** | **23 passed** |
| toate 5, după acest branch (`25f21c2`) | fp32 | **37 passed** |
| toate 5, după acest branch (`25f21c2`) | fp64 | **37 passed** |

**Concluzie:** circuitele QCBM și HQViT sunt corecte pe GPU. E prima verificare a lor pe `nvidia`; până acum
`notebooks/colab_gpu.ipynb` rula doar primele două fișiere de test. Cele 5 eșecuri vin din testele însele, care
fuseseră scrise doar pentru `qpp-cpu`, adică fp64:

| Test | Cauza în fp32 |
|---|---|
| `test_log_prob_normalizes_and_matches_samples[True]` | suma probabilităților e 1 ± 2·10⁻⁷, iar toleranța era 10⁻⁷ |
| `test_born_probs_gradient_is_exact` | diferențe finite cu ε = 10⁻⁶ sunt dominate de rotunjirea fp32 |
| `test_inverse_undoes_forward_any_norm[4]`, `[6]` | dus-întors cu eroare de ~2·10⁻⁶, iar toleranța era 10⁻⁶ |
| `test_register_evolve_autograd_matches_dense_reference` | `evolve` întoarce complex64 în fp32, iar referința presupunea complex128 |

**Ce s-a schimbat** (branch `gpu-gate-precision`; testele existente pe CPU rămân exact la fel):
- Cele 5 teste folosesc tiparul care exista deja în `test_hqvit_kernels.py`: toleranțe mai largi și pas mai mare
  la diferențe finite doar când `is_fp32()`. Pe `qpp-cpu` și pe `nvidia` fp64 se verifică la fel de strict ca înainte.
  **Verificat pe `nvidia` (T4, commit `25f21c2`):** toate cele 5 fișiere de test cuantic trec, cu 37 passed în fp32
  și 37 passed în fp64 (`Q2JEPA_CUDAQ_PRECISION=fp64`).
- Verificarea strictă a gradienților (diferențe finite cu ε = 10⁻⁶) rămâne doar în fp64. În fp32 testele sunt mai
  largi (ε = 10⁻², toleranță 2·10⁻²), deci antrenarea în fp32 se bazează pe faptul că aceleași circuite trec strict
  în fp64.
- Opțiune nouă `quantum.precision`: `default` (implicit; nu schimbă nimic), `fp32` sau `fp64`. Se aplică doar
  target-ului `nvidia`. Intră în identitatea checkpoint-ului doar când nu e `default`, deci checkpoint-urile vechi
  rămân valide.
- Testele citesc `Q2JEPA_CUDAQ_PRECISION` (de exemplu `fp64`), pe lângă `Q2JEPA_CUDAQ_TARGET`.
- În notebook, celula 4 rulează acum toate cele 5 fișiere de test cuantic, în fp32 și în fp64. Celula 6 rulează
  partea CPU a benchmark-ului de atenție doar la 16×16 și pe 4 imagini, pentru că circuitul de 21 de qubiți la
  32×32 nu terminase după 13 minute pe cele 2 vCPU-uri Colab.

## 2. Benchmark-ul de atenție (swap test)

`scripts/bench_attention.py`, patch-uri 4×4, ms pe imagine (după un apel de încălzire), `nvidia` în fp32. Rulat cu
16 imagini pe ambele coloane. Notebook-ul rulează acum coloana CPU doar la 16×16 și pe 4 imagini; ms pe imagine
rămâne comparabil, dar pentru exact acest tabel folosește `--images 16`.

| Rezoluție | Variantă | Qubiți | `qpp-cpu` | `nvidia` (T4) |
|---|---|---|---|---|
| 16×16 | factored | 4 | 122 | 108 |
| 16×16 | full | 17 | 4357 | 6,0 |
| 32×32 | factored | 4 | — (oprit) | 526 |
| 32×32 | full | 21 | — (oprit) | 35,4 |

- Circuitul full, de 17 qubiți, e de **~730 de ori** mai rapid pe T4 decât pe CPU. E rezultatul HPC pentru
  prezentare.
- Varianta factored e *mai lentă* pe GPU decât varianta full: înseamnă multe circuite mici de 4 qubiți, iar fiecare
  apel CUDA-Q costă ~7–8 ms de overhead (108 ms / 16 patch-uri, 526 ms / 64), oricât de mic e circuitul. Pe GPU,
  costul e numărul de apeluri, nu mărimea lor.

## 3. Gate-ul de timing (`scripts/bench_encoder_step.py`)

`+experiment=phase2` (64×64, patch-uri 8×8, `input_norm=l2`), batch 64, `--steps 1`. Coloanele sunt secunde pe
pas, iar estimarea pentru o rulare e
`(warmup_repr × repr + (iters − warmup_repr) × full + readout.iters × readout) / 3600`, cu pașii `pred` numărați ca
`full` (limită superioară).

**fp32 (implicit pe `nvidia`):**

| Encoder | Prior | repr s/it | full s/it | readout s/it | h/rulare | unități/rulare |
|---|---|---|---|---|---|---|
| CNN | Softmax-Categorical | 0,013 | 0,023 | 0,002 | 0,03 | 0,0 |
| CNN | Entangled-QCBM | 0,011 | 0,187 | 0,002 | 0,20 | 0,2 |
| ViT twin | Softmax-Categorical | 0,029 | 0,034 | 0,011 | 0,05 | 0,1 |
| ViT twin | Entangled-QCBM | 0,027 | 0,217 | 0,011 | 0,24 | 0,3 |
| **HQViT** | Softmax-Categorical | 24,98 | 21,65 | 0,63 | **24,9** | **28,7** |
| **HQViT** | Entangled-QCBM | 27,90 | 22,40 | 0,76 | **26,0** | **29,9** |

**fp64** (`cudaq.set_target("nvidia", option="fp64")`, rulat înainte să existe opțiunea `quantum.precision`):

| Encoder | Prior | repr s/it | full s/it | readout s/it | h/rulare | unități/rulare |
|---|---|---|---|---|---|---|
| HQViT | Softmax-Categorical | 46,73 | 49,89 | 1,12 | **56,5** | **65,0** |
| HQViT | Entangled-QCBM | 53,75 | 54,35 | 1,11 | **61,6** | **70,8** |

fp64 e de ~2,3–2,4 ori mai lent decât fp32.

**Ce înseamnă:**
- Toate rândurile clasice costă sub o unitate și sub 15 minute pe rulare.
- Un pas HQViT durează ~22–28 s. La ~800 de apeluri CUDA-Q pe pas și ~8 ms de overhead pe apel (§2), overhead-ul
  explică singur ~6–7 s; restul e simularea registrelor de ~20 de qubiți și conversiile între torch și CUDA-Q. Un GPU
  mai rapid ajută doar la a doua parte.
- QCBM adaugă aproape nimic la HQViT (~1 h pe rulare), pentru că pasul e dominat de encoder.
- `--steps 1` înseamnă un singur pas măsurat pe stadiu; variația dintre pași nu e cunoscută. `repr` > `full` la
  HQViT e probabil zgomot de măsurare.

## 4. Ce decide pentru grid

**Costul HQViT:** ~25 h și ~29 de unități pe rulare (T4, fp32). Decizia o ia echipa.

**Tipul de GPU:** de completat după L4, A100 și G4. Pe baza §2, un GPU mai rapid probabil reduce doar o parte
din pas.

**Precizia pentru grid: propunere fp32**, de confirmat de echipă.
- În fp64, o rulare HQViT trece de 36 h și de 60 de unități ; 6 rulări de grid ar costa ~390 de
  unități.
- Corectitudinea circuitelor e verificată în fp64 (§1). fp32 adaugă erori relative de ~10⁻⁶, mult sub zgomotul
  deja prezent în antrenare: sampling-ul QCBM cu shot-uri finite, flip-urile de 0,5% și SGD.
- Toate rândurile unui tabel folosesc aceeași precizie, pentru că face parte din platformă (F9). Se trece explicit
  în override-uri (`quantum.precision=fp32`), nu se lasă pe `default`.

**Dacă rulările HQViT trebuie scurtate:** fallback-ul din spec §6, batching de circuite (mai puține apeluri CUDA-Q),
atacă exact costul dominant de la §2–3.

## 5. Cum se reproduce

În `notebooks/colab_gpu.ipynb` pe un runtime T4: celulele 1, 2, 4 și 6, apoi

```bash
!cd /content/Q2-JEPA && python scripts/bench_encoder_step.py --target nvidia \
    --encoder cnn --encoder vit --encoder hqvit \
    --prior softmax_categorical --prior entangled_qcbm --steps 1 --cu-per-hour 1.15
# fp64, după acest branch:
!cd /content/Q2-JEPA && python scripts/bench_encoder_step.py --target nvidia --encoder hqvit \
    --prior softmax_categorical --prior entangled_qcbm --steps 1 --cu-per-hour 1.15 --set quantum.precision=fp64
```
