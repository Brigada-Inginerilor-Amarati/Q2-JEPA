# Sprint Colab: antrenare până joi, 27 noiembrie 2026, dimineața

**Scop:** tabelele de rezultate phase 2 pentru MovingMNIST (CNN, ViT twin, HQViT × Softmax-Categorical,
Entangled-QCBM × seed-urile 0–2) plus inferența IBM pe prior-ul Born antrenat, toate reproductibile din
override-uri, seed și SHA. HEP18 intră doar dacă se termină la timp.

**Resurse:** 3 conturi Google AI Pro (200 de unități Colab pe lună fiecare, ~600 în total), 8 zile.

**Referințe:** [phase-2 spec](../specs/2026-11-03-q2jepa-phase2-design.md) (§4 regulile F1–F11, §6 gate-ul
HQViT, §9 grid-ul, §10 ordinea), [dataset spec](../specs/2026-11-03-q2jepa-datasets-hep18-d3il.md),
[AGENTS.md](../../../AGENTS.md) §3 (reguli stricte) și §5 (Colab).

---

## 0. Ce înseamnă „gata” pe 27 noiembrie

În 8 zile nu se poate termina tot phase 2. Plan pe trei niveluri:

| Nivel | Ce include | Condiție |
|---|---|---|
| **M: obligatoriu** | Tuning F5 (3 scheme × 3 encodere, dev, seed 0) · grid MovingMNIST: 3 encodere × 2 priori × 3 seed-uri = **18 rulări** · tabel în `docs/results/` · inferență IBM pe un prior Born antrenat | gate-ul HQViT trece (§4, D2) |
| **S: dacă rămâne timp** | Ablația `pos_emb=none` (twin, HQViT) · Product/Frozen-QCBM · HEP18 cu CNN și twin × Softmax, Entangled-QCBM și Product-QCBM × 3 seed-uri | HEP18 integrat în `train.py` până luni 24 seara |
| **W: nu intră până pe 27** | D3IL · HQViT pe HEP18 (~36× mai multe cadre pe pas decât MovingMNIST) · sanity rows HEP18/D3IL cu encoderele din paper | — |

---

## 1. Roluri

| Cine | Rol | Livrează |
|---|---|---|
| **P1** (Daria, Windows) | Infrastructură GPU | C1, C3, C4, gate-ul și deciziile D1–D3, monitorizarea rulărilor |
| **P2** (Paul, Mac) | Evaluare, rezultate | C2, C5, C6 |
| **P3** (Doca, Linux) | HEP18 | C8: adaptorul din `train.py`, renderer, render check |
| **Nico** (Windows, fără unități Colab) | IBM, prezentare | C7, inferența pe QPU, prezentarea finală |
| **Alex** (Mac, fără unități Colab) | HEP18, review | Readout-ul și metricile HEP18 din C8, review spec-rules pe PR-uri |
| **P1, P2, P3** | Grid-ul | Fiecare rulează un seed: P1 → seed 0, P2 → seed 1, P3 → seed 2 |

Grid-ul e împărțit pe seed ca fiecare cont să aibă aceeași încărcare și aceleași rânduri HQViT.

**Mediu local.** CUDA-Q (`cuda-quantum-cu13==0.16.0`) are wheel doar pentru Linux și Mac cu Apple Silicon. Paul și
Doca lucrează direct. Alex direct dacă are chip Apple M…, altfel ca Nico. Nico (Windows) folosește GitHub Codespaces
din repo. Daria folosește WSL. Nimeni nu antrenează local; antrenarea e doar pe Colab.

---

## 2. Pregătire conturi (fiecare, miercuri 19, seara, ~15 min)

- [ ] AI Pro e activ, iar Colab afișează **compute units** în panoul *Resources*.
- [ ] Ai acces la `Brigada-Inginerilor-Amarati/Q2-JEPA`. Dacă repo-ul e privat: un token GitHub read-only pus în
      **Colab Secrets** (cheia `GITHUB_TOKEN`), niciodată scris în notebook.
- [ ] Pe Drive există `MyDrive/q2jepa/runs/`. P1 creează un folder comun `q2jepa-results/` partajat cu ceilalți:
      acolo merg doar `metrics*.json`, `timing.json`, `params.json` și `.hydra/config.yaml`, nu checkpoint-uri.
- [ ] P2 și P3 (cei care scriu cod): `uv sync`, `lefthook install`, iar `uv run pytest` trece local.
- [ ] Nico: un cont IBM Quantum (Open Plan gratuit, 10 minute QPU pe 28 de zile). Contul și token-ul le face
      persoana, nu un agent. Token-ul stă doar în variabila de mediu `QISKIT_IBM_TOKEN`.

---

## 3. Schimbări de cod

Fiecare schimbare: branch → PR → `uv run pytest` și `uv run ruff check .` trec → review `spec-rules-reviewer` → merge în `main`.

### C1. Benchmark pentru un pas de encoder (P1, miercuri 19)
**Fișiere:** `scripts/bench_encoder_step.py` (nou), `tests/test_bench.py` (un caz mic CNN pe `qpp-cpu`).
- `train.py` expune `train_step`, `build_models` și `resolve_device`. Benchmark-ul cronometrează exact funcția pe
  care o folosește `run()`. Refactorul e verificat: metricile rulărilor mici sunt identice byte cu byte.
- Cu setările `+experiment=phase2` (64×64, patch-uri 8×8, `input_norm=l2`), batch 64, pe `--target` (`nvidia` sau
  `qpp-cpu`), măsoară pentru fiecare combinație encoder × prior:
  - un `train_step` în stadiul `repr`;
  - un `train_step` în stadiul `full` (sampling din prior plus actualizarea lui; cel mai scump pas);
  - encodarea `no_grad` a lui `obs` și `fut` din stadiul de readout.
- Estimarea pentru o rulare:
  `ore = (warmup_repr × repr + (iters − warmup_repr) × full + readout.iters × readout) / 3600`.
  Pașii din stadiul `pred` sunt numărați ca `full`, deci estimarea e o limită superioară.
- Rezultatul gate-ului se trece în phase-2 spec §6, cu GPU-ul, SHA-ul și valorile măsurate.

### C2. Timp și număr de parametri scrise de `train.py` (P2, miercuri 19 → joi 20, 12:00; **intră în freeze**)
**Fișiere:** `src/q2jepa/train.py`, `tests/test_train.py`.
- `timing.json` în directorul rulării cu: timpul pe stadii (antrenare, readout, evaluare), secunde pe iterație,
  `torch.cuda.get_device_name()`, target-ul CUDA-Q, versiunile `cudaq` și `torch`. Versiunile contează pentru că
  fiecare cont își instalează mediul separat și două rânduri „T4” pot ajunge pe build-uri diferite (F9). Timpul se
  **adună peste reluări**, deci se ține și în `ckpt.pt`.
- `params.json` cu parametrii antrenabili împărțiți în cuantici (unghiurile θ ale HQViT, offset-urile și terminalul
  QCBM) și clasici (F8).
- Nimic din cele două nu intră în `_identity`, ca să nu invalideze checkpoint-urile.
- Contoarele de evaluări de circuit și shot-uri (F7, F11) sunt nivel **S**. Dacă nu sunt gata până la freeze, se
  calculează analitic din config în tabelul de rezultate, iar asta se notează în tabel.

### C3. Grid prin Hydra multirun și celula de notebook (P1, joi 20)
Nu se scrie un script nou de training (regula din AGENTS §3): se folosește `-m` din Hydra.
- **Verificat local:** `hydra.sweep.subdir='${encoder.name}-${prior.name}-s${seed}'` creează directoare
  deterministe, iar o a doua rulare pe un director terminat nu reantrenează: sare bucla, încarcă `readout.pt` și
  doar reevaluează. Adaugă un test în `tests/test_train.py` dacă nu există deja.
- **`notebooks/colab_gpu.ipynb`:** clone pe `GIT_REF` (tag-ul, implicit `grid-v1`), iar ultima celulă devine o celulă
  de parametri (`SEED`, `ENCODER`, `PRIORS`, schema LR) care rulează comanda multirun de mai jos. Logica rămâne în
  pachet; în notebook sunt doar parametri.
- Pentru HQViT se pune `trainer.ckpt_every=50` (nu face parte din identitate), ca o deconectare să coste maximum
  50 de iterații.

### C4. Freeze: tag `grid-v1` (P1, joi 20, ~14:00)
Condiții:
- [ ] C1–C3 sunt în `main`.
- [ ] `pytest` și `ruff` trec, iar review-ul spec-rules a fost făcut.
- [ ] Testele cuantice trec pe `nvidia` în Colab (AGENTS §5). Asta e și prima verificare a QCBM pe GPU.

După tag, nimic din ce atinge antrenarea MovingMNIST nu se mai schimbă (`src/q2jepa/` în afara fișierelor noi
HEP18, `conf/encoder|prior|trainer|quantum`, `conf/dataset/mmnist.yaml`). Un bug găsit după freeze înseamnă tag
`grid-v2` și rerularea rândurilor afectate. Rândurile clasice sunt ieftine; pentru HQViT se decide la caz.

### C5. `evaluate()`, Dice, readout-only (P2, vineri 21 → duminică 23)
**Fișiere:** `src/q2jepa/evaluate.py` (nou), `src/q2jepa/train.py` (îl apelează), `src/q2jepa/metrics.py`, teste.
- `evaluate(cfg, out_dir, sampler=None) -> dict`: încarcă `ckpt.pt` și `readout.pt` și calculează metricile
  existente, plus **Dice** ponderat cu probabilitatea direcției (§8.2) și metricile **readout-only** (F10).
- **Test de regresie:** pe o rulare mică, metricile de la `evaluate()` sunt identice cu cele de la `grid-v1`. Doar
  așa se poate schimba `train.py` după freeze.
- La final se reevaluează toate checkpoint-urile din grid, fără reantrenare. Rândurile din tabel trec atunci două
  SHA-uri: unul pentru antrenare (`grid-v1`) și unul pentru evaluare.
- Numerele de la C5 **înlocuiesc** metricile de test scrise de `grid-v1`. Nu se alege nimic între cele două
  evaluări (F5: pe test nu se face selecție).
- Argumentul `sampler` e punctul în care se conectează IBM (C7).

### C6. Helper pentru tabelul de rezultate (P2, sâmbătă 22)
**Fișiere:** `scripts/results_table.py` (nou) și un test.
- Citește directoarele de rulări și scrie un tabel markdown cu media ± s.d. de eșantion și valorile per seed
  pentru coverage, validity, TV, minADE, ES (și Dice și readout-only după C5), plus parametri, timp, GPU, SHA și
  override-uri.

### C7. Inferență IBM (Nico, vineri 21 → luni 24)
**Fișiere:** `scripts/eval_hardware.py`, extra-ul `[ibm]` în `pyproject.toml`, `tests/test_born_equivalence.py`
(probabilitățile Born din CUDA-Q și din Qiskit coincid, **inclusiv ordinea biților**: Qiskit e little-endian),
`tests/test_eval_hardware.py` (cu sampler simulat, fără rețea).
- Ia condiționerul antrenat, construiește cele 80 de circuite (unul pe observație), le trimite într-un singur job
  `SamplerV2` și calculează metricile cu `evaluate(..., sampler=...)`. Le compară apoi cu sampling-ul simulat pe
  `nvidia`.
- Prima țintă e checkpoint-ul `cnn-entangled_qcbm-s0`, gata de vineri. A doua, dacă e gata miercuri,
  `hqvit-entangled_qcbm-s0`.
- Dacă job-ul stă la coadă după termen, se raportează ca în așteptare. Un fake backend nu se raportează niciodată
  drept hardware.

### C8. HEP18 integrat în `train.py` (P3, miercuri 19 → luni 24, 18:00)
Întâi un plan scurt: `docs/superpowers/plans/2026-11-19-hep18-training.md`. Un benchmark nou înseamnă un modul
`data/`, unul `readout/`, un `conf/dataset/*.yaml` și metrici; nu un script nou.
- **Renderer:** dataset-ul HEP18 conține doar stări. Imaginile le produce `render_agent_clip` din `data/hep18.py`, pe
  device-ul batch-ului, la fiecare pas. Toate encoderele primesc aceleași imagini.
- **Train:** un adaptor pe dataset în `train.py`, cu:
  - encoder pe fiecare mașină, cu ponderi comune și `encoder.d=16`, iar cele 4 ieșiri concatenate dau 64;
  - 18 ținte pe observație;
  - biți locali pe agent (n=12, câte 3 pe mașină, cu mască);
  - predictor rezidual fără normalizare la ieșire;
  - distanța ρ netezită (β=0.5, ε=1e-4) și regularizarea de schimbare λΔ=2/3;
  - setări: 6000 de iterații, batch 32, readout 16000 × 128, b/B 8/32, r/G 12/60, 512 shot-uri (spec §8.3).
- **Fișiere (Alex):** `src/q2jepa/readout/hep18.py`; în `metrics.py`: validitate comună, coverage din 18, TV, minADE, ES,
  verificarea cu mașinile amestecate, validitate locală.
- **Render check:** σ și coloana din `data/hep18.py` sunt provizorii. Diagnosticul readout-only trebuie să treacă
  **și pe CNN, și pe ViT twin** (dataset spec §8). Apoi se fixează valorile și se rulează `scripts/render_examples.py`
  pentru noul `encoder.input_scale`. Dacă nicio valoare nu trece pe twin, se oprește lucrul și se discută în echipă.
- Dacă PR-ul nu e în `main` luni la 18:00, HEP18 trece la nivelul **W**.

---

## 4. Gate-ul și deciziile

| # | Când | Cine | Decizie |
|---|---|---|---|
| D1 | joi 20, dimineața | P1 | **Tipul de GPU** pentru toate tabelele: T4 sau L4. Se alege cel cu mai puține unități pe o rulare HQViT. La diferență mică se alege T4, care e mai ușor de obținut. Toți trei folosesc același tip (F9). |
| D2 | joi 20, dimineața | P1 | **Nivelul HQViT**, din costul estimat C (unități pe rulare) și durata H (ore pe rulare) de la C1 |
| D3 | vineri 21 (CNN, ViT) / după tuning (HQViT) | P1 | **Schema LR câștigătoare** pentru fiecare encoder: cel mai bun coverage pe dev, iar la egalitate decide TV (F5) |
| D4 | luni 24, seara | toți | **HEP18: intră sau nu** |
| D5 | marți 25 | P2 | Dacă inferența IBM pe HQViT mai încape |

**D2, pe niveluri:**
- **Verde** (H ≤ 16 h și C ≤ 35): planul întreg: tuning HQViT (3 rulări), grid HQViT (6), ablație (3).
- **Galben** (H ≤ 36 h și C ≤ 60): HQViT nu primește tuning propriu și folosește schema câștigătoare a twin-ului.
  E o deviație de la F5. E conservatoare doar pentru concluziile favorabile HQViT: un rezultat la egalitate sau
  negativ poate veni din LR-ul netunat, iar tabelul trebuie să spună asta. Grid HQViT 6 rulări, fără ablație HQViT.
- **Roșu** (altfel): fallback-urile din spec §6, în ordine:
  1. batching de circuite într-un singur apel CUDA-Q sau `nvidia-mqpu` (nu schimbă modelul);
  2. mai puține blocuri, aplicate **identic și twin-ului** (F4). Twin-ul se retunează pe dev (D3);
  3. batch mai mic, aplicat **tuturor rândurilor din tabel**, inclusiv CNN și comparațiile de prior (F1: batch-ul
     e o setare de trainer). Altfel, rândurile cu alt batch merg într-un tabel separat.

  Gate-ul se reface vineri 21. Dacă e tot roșu sâmbătă 22: HQViT doar pe seed 0, marcat în tabel „preliminar, nu respectă
  F6”.

**Bugetul de unități pe cont (200):** rulările clasice costă ~1–2 unități fiecare pe T4 (se confirmă la C1).
Păstrați **≥ 40 de unități rezervă** pe cont pentru reluări după crash-uri.

| Cont | M (Verde) | Estimare |
|---|---|---|
| P1 | tuning CNN + ViT (6) · tuning HQViT s1 · seed 0: 4 clasice + 2 HQViT · IBM | ~10 + 3C |
| P2 | tuning HQViT s2 · seed 1: 4 clasice + 2 HQViT | ~6 + 3C |
| P3 | tuning HQViT s3 · seed 2: 4 clasice + 2 HQViT · HEP18 dacă intră | ~6 + 3C (+ HEP18) |

---

## 5. Calendar

| Zi | P1 | P2 | P3 | Rulează pe GPU |
|---|---|---|---|---|
| **Mi 19** (seara) | C1 · notebook pe T4 cu Drive (teste `nvidia`, inclusiv QCBM) · C1 pe T4 și L4 | C2 | plan HEP18, C8 | P1: verificare + gate |
| **Jo 20** | D1, D2 · C3 · **14:00 freeze `grid-v1`** · pornește tuning CNN s1–s3 + ViT s1–s3 | termină C2 (până la 12:00) | C8 | tuning clasic (P1); tuning HQViT, câte o schemă pe cont (Verde) |
| **Vi 21** | D3 pentru CNN, ViT · grid clasic seed 0 | C5 · grid clasic seed 1 | C8 · grid clasic seed 2 | grid clasic, 4 rulări pe cont. Pe Galben pornește și grid-ul HQViT |
| **Sâ 22** | monitorizare · D3 HQViT (Verde) → pornește grid HQViT | C6 · prim tabel clasic | C8 | grid HQViT |
| **Du 23** | monitorizare | C7 | C8 · diagnostic readout-only | grid HQViT |
| **Lu 24** | IBM pe `cnn-entangled_qcbm-s0` (cu P2) | C7 gata | PR HEP18 → review → **D4 la 18:00** | grid HQViT |
| **Ma 25** | ablație dacă mai sunt unități · **23:59: ultima oră la care se poate porni o rulare care intră în tabele** | C5 merge | rulări HEP18 (dacă D4 = da), tag `hep18-v1` | HQViT termină · HEP18 |
| **Mi 26** | reevaluare C5 pe toate checkpoint-urile · IBM pe HQViT (D5) | `docs/results/2026-11-mmnist-phase2.md` (+ `-hep18.md`) | actualizează tabelul de stare din AGENTS.md și README | doar ce se termină |
| **Jo 27** (dimineața) | `pytest`, `ruff` · tag `results-2026-11-27` | verifică fiecare rând: override-uri, seed, SHA, GPU | — | nimic |

Nico și Alex nu apar în tabel: Nico preia de la P2 tot ce ține de IBM (C7 și rulările de luni și miercuri) și face
prezentarea; Alex preia de la P3 readout-ul și metricile HEP18 și face review-ul spec-rules pe C2, C3 (joi până la 13:00),
C5 și HEP18 (luni până la 16:00).

Rulările care nu ajung la toate iterațiile până miercuri seara **nu intră în tabel** (F7: aceleași iterații pentru
toate rândurile). Se menționează doar ca „neterminat”.

---

## 6. Cum se rulează în Colab

În toate celulele: `+experiment=phase2 quantum=gpu quantum.precision=fp32 trainer.device=cuda`, același tip de GPU, tag-ul `grid-v1`.

**Tuning (dev, seed 0):**
```bash
python -m q2jepa.train -m +experiment=phase2 quantum=gpu quantum.precision=fp32 trainer.device=cuda \
  encoder=cnn,vit +tune=s1,s2,s3 seed=0 \
  hydra.sweep.dir=/content/drive/MyDrive/q2jepa/runs/mmnist-tune \
  'hydra.sweep.subdir=${encoder.name}-lr${trainer.lr_enc}-${trainer.lr_enc_later}-s${seed}'
```

**Grid (un encoder pe celulă, pentru că fiecare are schema lui de la D3; exemplu pentru seed 1):**
```bash
python -m q2jepa.train -m +experiment=phase2 quantum=gpu quantum.precision=fp32 trainer.device=cuda \
  encoder=vit prior=softmax_categorical,entangled_qcbm seed=1 \
  trainer.lr_enc=<D3> trainer.lr_enc_later=<D3> \
  hydra.sweep.dir=/content/drive/MyDrive/q2jepa/runs/mmnist \
  'hydra.sweep.subdir=${encoder.name}-${prior.name}-s${seed}'
```

**HQViT:** o singură rulare pe sesiune, cu `trainer.ckpt_every=50`. Dacă sesiunea pică, rulați aceeași celulă din
nou; reluarea vine din `ckpt.pt`. Dacă Colab vă permite două sesiuni GPU în paralel pe același cont, cele 2 rulări
HQViT ale unui seed pot merge simultan.

**După fiecare sesiune:**
- [ ] Copiați `metrics*.json`, `timing.json`, `params.json` și `.hydra/config.yaml` în `q2jepa-results/<run>/`.
- [ ] *Runtime → Disconnect and delete runtime*. Unitățile se consumă cât timp runtime-ul e conectat, chiar dacă nu
      rulează nimic.
- [ ] Notați în canalul echipei: rularea, iterația la care a ajuns și unitățile rămase.

---

## 7. Checklist de reguli (orice rulare care intră într-un tabel)

- [ ] `+experiment=phase2` pe toate rândurile ViT și HQViT, și pe CNN în tabelele phase 2 (F2).
- [ ] Același tip de GPU pe tot tabelul (F9), trecut în fiecare rând.
- [ ] Seed-urile 0–2 (F6). Dacă diferența dintre două rânduri e sub o s.d., se adaugă seed-urile 3–4 înainte de
      orice concluzie. Dacă nu mai sunt unități, rândul spune „fără concluzie (sub 1 s.d.)”, fără câștigător.
- [ ] Tuning-ul se face numai pe `dev`. Pe `test` se evaluează o singură dată, pe checkpoint-ul final (F5).
- [ ] Între rândurile comparate diferă doar componenta testată (F1). HQViT se compară cu twin-ul (F4).
- [ ] În notebook nu se modifică decât parametrii. Orice altă schimbare se face în cod, cu commit și tag nou.
- [ ] Niciun token (GitHub, IBM) în repo sau în notebook.

---

## 8. Riscuri

| Risc | Ce facem |
|---|---|
| HQViT prea lent | Nivelurile D2. Fallback-urile se aplică identic și twin-ului. |
| Nu se găsește GPU-ul ales la D1 | La D1, alegeți T4 dacă L4 nu e clar mai ieftin. Dacă GPU-ul lipsește > 2 h, așteptați. Nu schimbați GPU-ul în mijlocul unui tabel. |
| Deconectări | Reluare din `ckpt.pt`. `ckpt_every=50` la HQViT. Folosiți Drive, niciodată directorul de fallback. |
| QCBM pe `nvidia` pică testele | Se repară în `quantum/` înainte de freeze. Freeze-ul se mută; tuning-ul clasic poate porni oricum, separat. |
| Bug după freeze | `grid-v2` și rerularea rândurilor afectate. Ce s-a schimbat se notează în tabel. |
| Coadă IBM lungă | Job-ul se trimite luni. Dacă nu se termină până miercuri, apare în raport ca în așteptare. |
| HEP18 nu e gata | D4 luni seara. Dacă răspunsul e nu, HEP18 trece la nivelul W, fără rulări pe jumătate. |
