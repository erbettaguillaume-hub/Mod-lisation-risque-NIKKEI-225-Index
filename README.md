# 📉 Modélisation GARCH du Nikkei 225 — Prévision du risque de marché (VaR & Expected Shortfall)

Ce projet implémente une **chaîne complète de modélisation et de backtesting du risque** sur les rendements du **Nikkei 225**, à partir d’un modèle **AR–GARCH(1,1)** avec innovations **Student-t**.  
L’objectif est de **prévoir la volatilité conditionnelle** et de **quantifier les pertes extrêmes** via **Value-at-Risk (VaR)** et **Expected Shortfall (ES)**, puis de **valider la qualité prédictive** par **backtesting**.

---

## 🎯 Objectifs

### 1) Modélisation du risque (volatilité conditionnelle)
- Transformer une série de prix en **log-rendements**.
- Vérifier la **stationnarité** et diagnostiquer les **dépendances** (autocorrélation / hétéroscédasticité conditionnelle).
- Estimer des modèles **GARCH** avec **distribution Student-t** (queue épaisse) pour mieux capturer les extrêmes.

### 2) Mesure des pertes extrêmes (VaR / ES)
- Calculer des prévisions **à 1 jour** de :
  - moyenne conditionnelle \( \mu_{t+1} \)
  - volatilité conditionnelle \( \sigma_{t+1} \)
  - **VaR** et **ES** au niveau \( \alpha \)

### 3) Validation out-of-sample (backtesting)
- Mettre en place un backtest **rolling / expanding** :
  - **ré-estimation** du modèle à chaque date de test
  - comptage des **violations de VaR** (hits)
- Vérifier la cohérence empirique : **violations observées** vs **violations attendues** \(\approx \alpha \times N\).

---

## 🧠 Méthodologie (pipeline)

### A) Data engineering (robustesse & qualité)
- Chargement du fichier `NIKKEI.csv` et nettoyage :
  - parsing dates, conversion numérique, tri temporel, suppression doublons
  - contraintes : **prix > 0** et **N ≥ 120** observations minimum

### B) Construction des rendements
- Log-rendements :
\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

### C) Diagnostics statistiques
- **Stationnarité** :
  - ADF (H0 : racine unitaire)
  - KPSS (H0 : stationnarité)
- **Dépendance / hétéroscédasticité** :
  - Ljung-Box sur \(r_t\) puis \(r_t^2\)
  - ARCH-LM (détection d’effets ARCH)

### D) Estimation GARCH (Student-t)
Modèle estimé (forme générale) :
\[
r_t = \mu + \sum_{i=1}^{L}\phi_i r_{t-i} + \varepsilon_t,\quad \varepsilon_t=\sigma_t z_t,\quad z_t\sim t_\nu
\]
\[
\sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{q}\beta_j \sigma_{t-j}^2
\]

- Comparaison de plusieurs spécifications **AR(lag)-GARCH(1,1)-t** via **AIC/BIC**
- Estimation par **Maximum Likelihood** (`arch`)

### E) Diagnostics de résidus (qualité d’ajustement)
Sur les résidus standardisés \(z_t=\hat{\varepsilon}_t/\hat{\sigma}_t\) :
- Ljung-Box sur \(z_t\) et \(z_t^2\)
- ARCH-LM (absence d’effets ARCH résiduels)
- Test d’asymétrie **Engle–Ng** (sign bias / size bias)

### F) Prévisions VaR / ES (Student-t standardisée)
- Quantile et ES de queue gauche pour une Student-t **standardisée variance 1**
- Prévision à 1 jour :
\[
\text{VaR}_{t+1,\alpha} = -(\mu_{t+1} + \sigma_{t+1}q_{\alpha})
\quad;\quad
\text{ES}_{t+1,\alpha} = -(\mu_{t+1} + \sigma_{t+1}ES_{\alpha})
\]
Le signe `-` convertit la queue gauche des rendements en **mesure de perte positive**.

### G) Backtesting rolling / expanding
- Ré-estimation du modèle à chaque date de test
- **Hit VaR** :
\[
\text{hit}_{t+1}=\mathbf{1}\left(r_{t+1} < -\text{VaR}_{t+1,\alpha}\right)
\]
- Traçage conjoint : rendements réalisés vs seuils VaR/ES.

---

## ✅ Résultats (exécution du notebook sur l’échantillon fourni)

*(Les chiffres ci-dessous proviennent de la dernière exécution enregistrée dans le notebook.)*

- **Stationnarité** : ADF/KPSS cohérents → rendements stationnaires.
- **Hétéroscédasticité conditionnelle** :
  - Ljung-Box sur \(r_t^2\) significatif et ARCH-LM significatif → présence d’effets ARCH (justifie un GARCH).
- **Sélection de modèle** :
  - AR lags testés (0,1,2), sélection selon BIC → **Constant-GARCH(1,1)-t** retenu.
  - Queue épaisse : paramètre **\(\nu \approx 5.78\)** (Student-t).
- **Diagnostics résiduels** :
  - Ljung-Box sur résidus et résidus² non significatifs
  - ARCH-LM non significatif → pas d’ARCH résiduel détecté
  - Engle–Ng : pas d’asymétrie significative détectée sur l’échantillon
- **Backtesting VaR (α = 5%)** :
  - Nombre d’observations backtest : **124**
  - Violations observées : **6**
  - Violations attendues : **~ 6.20** (\(\alpha \times N\))

➡️ Lecture “risk” : la fréquence de violations est **cohérente** avec le niveau de risque ciblé à 5% sur cet échantillon.

---

## ⚙️ Paramètres (config)
Paramètres principaux (en tête du notebook) :
- `ALPHA = 0.05` (niveau VaR/ES)
- `BACKTEST_START = 120` (début backtest)
- `ROLLING_WINDOW = None` (expanding par défaut, rolling si entier)
- `MEAN_LAGS = 0` (constant mean, AR si > 0)
- `P_ORDER = 1`, `Q_ORDER = 1`
- `SCALE_TO_PCT = True` (mise à l’échelle *100 pour stabilité numérique*)

---

## 📦 Prérequis & dépendances
Librairies principales :
- `pandas`, `numpy`, `matplotlib`
- `statsmodels` (ADF, KPSS, Ljung-Box, ARCH-LM, OLS)
- `arch` (estimation GARCH)
- `scipy` (Student-t)

Installation (exemple) :
```bash
pip install numpy pandas matplotlib statsmodels arch scipy
