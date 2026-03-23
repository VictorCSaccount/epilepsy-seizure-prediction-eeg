# Predictia Crizelor Epileptice — Pipeline Adaptive Baseline

Un pipeline de machine learning specific fiecarui pacient pentru predictia
crizelor epileptice din inregistrari EEG de scalp. Dezvoltat ca proiect de
semestru pentru cursul de Sisteme Bazate pe Cunostere, Facultatea de Automatica
si Calculatoare, Universitatea Tehnica din Cluj-Napoca (2026).

Sistemul a fost implementat si testat in Google Colab, folosind Google Drive
ca stocare a datelor.

---

## Problema abordata

Epilepsia afecteaza aproximativ 50 de milioane de persoane la nivel mondial.
Aproximativ 30% dintre pacienti nu raspund la tratamentul medicamentos. Un sistem
care poate identifica starea preictala (perioada de dinainte de criza) ar permite
pacientilor sa ia masuri preventive, sa alerteze ingrijitorii sau sa activeze
dispozitive de neurostimulare, reducand riscul de vatamare.

Obiectivul nu este detectia crizei (identificarea unei crize deja in desfasurare),
ci predictia — identificarea tranzitiei de la starea interictala la starea preictala
inainte de debutul clinic.

---

## Arhitectura sistemului

```
Fisier EDF
   |
   v
Preprocesare (notch, bandpass, CAR, detrend, diferentiere)
   |
   +-- Banda Alpha (8-13 Hz) --> features --> Random Forest (alpha)  --+
   |                                                                    |
   +-- Banda Beta  (13-30 Hz) --> features --> Random Forest (beta)  --+--> Vot --> Alarma
   |                                                                    |
   +-- Banda Broad (1-45 Hz)  --> features --> Random Forest (broad) --+
```

Trei clasificatori Random Forest independenti sunt antrenati per pacient, unul
pentru fiecare banda de frecventa. Probabilitatile lor de iesire sunt mediate
intr-un singur semnal de vot. O detectie este confirmata doar daca votul netezit
depaseste un prag si persista un timp minim, ceea ce reduce substantial alarmele false.

---

## Pipeline de procesare a semnalului

Urmatorii pasi sunt aplicati fiecarui segment EEG brut inainte de extragerea trasaturilor:

1. Filtru notch la 50 Hz — elimina interferenta retelei electrice
2. Filtru band-pass FIR 1-45 Hz (fereastra Hamming) — elimina deriva lenta si
   artefactele musculare de frecventa inalta
3. Common Average Reference (CAR) — scade media tuturor canalelor din fiecare canal,
   eliminand zgomotul comun tuturor electrozilor
4. Detrending liniar — elimina deriva reziduala in cadrul fiecarei ferestre
5. Diferentiere temporala de ordinul 1 — amplifica spike-urile epileptice si
   suprima activitatea de fond lenta

---

## Vectorul de trasaturi

Fiecare fereastra de 4 secunde este reprezentata prin patru valori scalare:

| Trasatura | Descriere |
|---|---|
| log Line Length | Diferenta medie absoluta de amplitudine intre esantioane consecutive. Creste brusc la debutul crizei, datorita cresterii simultane a amplitudinii si frecventei. |
| log Energy | Puterea medie a semnalului pe toate canalele. |
| Indice de Focalitate | max(energie canal) / medie(energie canal). Aproape 1 pentru zgomot difuz; mare pentru descarcari epileptice spatiale localizate. |
| Sincronizare Imaginara de Faza | Valoarea absoluta medie a partii imaginare a vectorului de sincronizare de faza (Transformata Hilbert). Imuna la conductia de volum: zgomotul instantaneu are Im ≈ 0; interactiunea neuronala reala are Im > 0. |

Transformarile logaritmice pe Line Length si Energy linearizeaza distributiile
acestora si imbunatatesc performanta clasificatorului.

---

## Strategia adaptiva de split

O provocare cheie este ca timpul disponibil de inregistrare inainte de o criza variaza
mult intre fisiere. Pipeline-ul gestioneaza acest lucru adaptiv:

- Daca sunt disponibile mai mult de 20 de minute inainte de debut: ultimele 15 minute
  sunt etichetate preictal, iar cele 10 minute dinainte sunt etichetate baseline.
- Daca sunt disponibile mai putin de 20 de minute: timpul este impartit in doua,
  prima jumatate ca baseline si a doua jumatate ca preictal.

Aceasta evita eliminarea inregistrarilor scurte, care ar reduce setul de antrenare
pentru pacientii cu putine crize.

---

## Clasificare si sistem de vot

Fiecare model de banda este un Random Forest cu urmatoarea configuratie:

| Parametru | Valoare | Motivatie |
|---|---|---|
| n_estimators | 50 | Echilibru intre viteza si stabilitate |
| max_depth | 6 | Previne memorarea zgomotului din baseline |
| class_weight | balanced | Compenseaza dezechilibrul mare dintre ferestele de baseline si preictal (ferestele preictale reprezinta ~5-6% din date) |

Trasaturile sunt scalate cu RobustScaler inainte de antrenare si testare,
reducand influenta valorilor extreme din artefactele de miscare.

Votul final este media aritmetica a probabilitatilor celor trei bande:

    vot = (p_alpha + p_beta + p_broad) / 3

Un vot netezit (medie mobila pe 5 ferestre) peste 0.60, care persista cel putin
8 secunde (2 ferestre consecutive la 50% overlap), confirma o detectie.

Aceasta abordare exploateaza proprietatile complementare ale benzilor: modelul
Alpha este sensibil dar produce alarme false in timpul relaxarii; modelul Beta
este mai specific dar poate rata modificarile preictale subtile; modelul Broad
adauga stabilitate. Necesitarea acordului intre bande reduce substantial
fals-pozitivele.

---

## Metodologia de evaluare

Se aplica validare incrucisata leave-one-seizure-out per pacient. Pentru fiecare
criza, modelul este antrenat pe toate celelalte crize disponibile ale aceluiasi pacient,
apoi testat pe cea retinuta.

Metrici:

- Sensibilitate: proportia crizelor detectate in fereastra preictala
- Specificitate: proportia minutelor de baseline fara alarma falsa
- FAR (Rata Alarmelor False): numarul de alarme false pe ora de baseline

---

## Seturi de date

### Siena Scalp EEG
- 14 pacienti adulti, 9 barbati si 5 femei, varste 20-71 ani
- Frecventa de esantionare: 512 Hz (redusa la 256 Hz pentru compatibilitate)
- Echipamente: amplificatoare EB Neuro si Natus Quantum LTM
- Adnotari conform clasificarii ILAE (IAS, FBTC)
- Sursa: Universitatea din Siena, Unitatea de Neurologie

### CHB-MIT Scalp EEG
- 23 pacienti pediatrici, 18 fete si 5 baieti, varste 1.5-22 ani
- Frecventa de esantionare: 256 Hz
- Colectata la Children's Hospital Boston
- Montaj bipolar standard 10-20

Ambele seturi de date folosesc fisiere EDF (European Data Format) si necesita
fisiere CSV de adnotari care mapeaza fiecare criza la fisierul de inregistrare
si timpul de debut.

---

## Rezultate

| Metrica | Valoare |
|---|---|
| Sensibilitate | 83.77% |
| Specificitate | 56.20% |
| FAR | < 26 / ora |

Rezultatele pe Siena au fost superioare celor pe CHB-MIT, consistent cu
frecventa de esantionare mai mare care ofera rezolutie spectrala mai buna
si calitate mai curata a semnalului.

---

## Instalare si utilizare

### Cerinte

Proiectul a fost dezvoltat si testat in Google Colab. Pentru rulare locala,
instaleaza pachetele necesare:

```bash
pip install mne numpy pandas scipy scikit-learn
```

### Pregatirea datelor

1. Descarca seturile de date de pe PhysioNet:
   - CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
   - Siena:   https://physionet.org/content/siena-scalp-eeg/1.0.0/

2. Pregateste doua fisiere CSV de adnotari cu urmatoarele coloane:

```
patient, test_number, seizure_id, reg_start_time, seizure_start_time
```

unde `reg_start_time` si `seizure_start_time` sunt in formatul `HH:MM:SS`.

### Utilizare in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Actualizeaza BASE_PATH in seizure_prediction.py sa corespunda structurii
# tale din Drive, apoi ruleaza:
run_pipeline('SIENA')
run_pipeline('MIT')
```

### Utilizare locala

Actualizeaza constantele de cale din partea de sus a fisierului si ruleaza:

```bash
python seizure_prediction.py
```

---

## Limitari cunoscute

- Artefactele musculare masive (masticatie, miscare faciala) pot produce valori
  ridicate de Line Length care seamana cu activitatea de criza daca nu sunt
  filtrate adecvat.
- Performanta modelului depinde de calitatea segmentului de baseline folosit
  pentru antrenare. Un baseline contaminat cu micro-descarcari poate reduce
  specificitatea.
- Definitia ferestrei preictale este aproximativa. O inregistrare in care
  pacientul era deja agitat inainte de debut poate reduce sensibilitatea.
- Specificitatea pe CHB-MIT este mai mica decat pe Siena, probabil din cauza
  frecventei de esantionare mai mici si a variabilitatii mai mari a baseline-ului
  in inregistrarile pediatrice.

---

## Directii de dezvoltare

- Inlocuirea netezirii prin medie mobila cu o Medie Mobila Exponentiala (EMA)
  cu viteza de adaptare variabila (mai rapida in starea de veghe activa, mai lenta
  in somn).
- Adaugarea indicelui Weighted Phase Lag Index (wPLI) ca trasatura suplimentara
  pentru imunitate sporita la zgomot.
- Hibridizare cu o retea CNN-LSTM pentru a capta patternuri temporale pe termen
  lung in ferestre preictale de 30 de minute.
- Portarea pipeline-ului pe un sistem embedded (ESP32 sau Raspberry Pi) pentru
  inferenta in timp real pe un dispozitiv EEG purtabil.

---

## Dependente

| Pachet | Scop |
|---|---|
| mne | Incarcarea fisierelor EDF, filtrare, gestionarea montajului |
| numpy | Operatii cu matrice, ferestre glisante |
| pandas | Incarcarea adnotarilor, agregarea rezultatelor |
| scipy | Detrending, Transformata Hilbert |
| scikit-learn | Random Forest, RobustScaler |

---

## Referinte

1. Commission on Classification and Terminology of the ILAE, Epilepsia, 1989.
2. WHO Epilepsy Fact Sheet, 2024.
3. Eadie M.J., Expert Review of Neurotherapeutics, 2012.
4. Detti et al., Processes, 2020.
5. Detti et al., IEEE Trans. Biomed. Eng., 2019.
6. Paszkiel S., Analysis and Classification of EEG Signals for BCI, Springer, 2020.
7. Wolpaw et al., Clinical Neurophysiology, 2002.

---

## Licenta

MIT
