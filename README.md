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
  - moyenne conditionnelle 
  - volatilité conditionnelle 
  - **VaR** et **ES** au niveau alpha
    
### 3) Validation out-of-sample (backtesting)
- Mettre en place un backtest **rolling / expanding** :
  - **ré-estimation** du modèle à chaque date de test
  - comptage des **violations de VaR** (hits)
- Vérifier la cohérence empirique : **violations observées** vs **violations attendues**

---

## 🧠 Méthodologie (pipeline)

### A) Data engineering (robustesse & qualité)
- Chargement du fichier `NIKKEI.csv` et nettoyage :
  - parsing dates, conversion numérique, tri temporel, suppression doublons
  - contraintes : **prix > 0** et **N ≥ 120** observations minimum

### B) Construction des rendements
- Log-rendements :

### C) Diagnostics statistiques
- **Stationnarité** :
  - ADF (H0 : racine unitaire)
  - KPSS (H0 : stationnarité)
- **Dépendance / hétéroscédasticité** :
  - Ljung-Box sur r puis r^2
  - ARCH-LM (détection d’effets ARCH)

### D) Estimation GARCH (Student-t)
Modèle estimé (forme générale) :
  - AR(0)-GARCH(1,1)
  - Pas d'effets d'asymétrie significatifs

- Comparaison de plusieurs spécifications **AR(lag)-GARCH(1,1)-t** via **AIC/BIC**
- Estimation par **Maximum Likelihood** ('arch`)

### E) Diagnostics de résidus (qualité d’ajustement)
Sur les résidus standardisés :
- Ljung-Box sur \(z_t\) et \(z_t^2\)
- ARCH-LM (absence d’effets ARCH résiduels)

### F) Prévisions VaR / ES (Student-t standardisée)
- Quantile et ES de queue gauche pour une Student-t **standardisée variance 1**
- Prévision à 1 jour

### G) Backtesting rolling / expanding
- Ré-estimation du modèle à chaque date de test
- **Hit VaR** :
- Traçage conjoint : rendements réalisés vs seuils VaR/ES.

## 📦 Prérequis & dépendances
Librairies principales :
- `pandas`, `numpy`, `matplotlib`
- `statsmodels` (ADF, KPSS, Ljung-Box, ARCH-LM, OLS)
- `arch` (estimation GARCH)
- `scipy` (Student-t)

Installation (exemple) :
```bash
pip install numpy pandas matplotlib statsmodels arch scipy
