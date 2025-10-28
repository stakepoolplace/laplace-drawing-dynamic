(venv) ericmarchand@macbookair-1 ~/laplace % python 10-laplaceNN-vs-feedforwardNN.py
================================================================================
COMPARAISON : RÉSEAUX D'OSCILLATEURS vs PERCEPTRONS CLASSIQUES
================================================================================

================================================================================
TÂCHE 1/8
================================================================================
📋 Sinusoïde pure (50 Hz)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 65.94 dB | MSE=0.000000                                                                                                                                                                  
SNR MLP          : -0.01 dB | MSE=0.501198
➡️  Gagnant : Oscillateurs (ΔSNR = +65.95 dB)

================================================================================
TÂCHE 2/8
================================================================================
📋 Chirp (10→200 Hz)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 30.70 dB | MSE=0.000425                                                                                                                                                                  
SNR MLP          : 0.25 dB | MSE=0.470843
➡️  Gagnant : Oscillateurs (ΔSNR = +30.45 dB)

================================================================================
TÂCHE 3/8
================================================================================
📋 AM (100 Hz ± 5 Hz)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 48.59 dB | MSE=0.000008                                                                                                                                                                  
SNR MLP          : -0.02 dB | MSE=0.564749
➡️  Gagnant : Oscillateurs (ΔSNR = +48.62 dB)

================================================================================
TÂCHE 4/8
================================================================================
📋 Signal composite [20.0, 60.0, 150.0] Hz
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 36.49 dB | MSE=0.000037                                                                                                                                                                  
SNR MLP          : 0.01 dB | MSE=0.166098
➡️  Gagnant : Oscillateurs (ΔSNR = +36.48 dB)

================================================================================
TÂCHE 5/8
================================================================================
📋 Onde carrée (approx.)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 41.19 dB | MSE=0.000073                                                                                                                                                                  
SNR MLP          : 1.18 dB | MSE=0.729809
➡️  Gagnant : Oscillateurs (ΔSNR = +40.00 dB)

================================================================================
TÂCHE 6/8
================================================================================
📋 Onde en dents de scie (approx.)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 31.99 dB | MSE=0.000197                                                                                                                                                                  
SNR MLP          : 0.51 dB | MSE=0.276901
➡️  Gagnant : Oscillateurs (ΔSNR = +31.47 dB)

================================================================================
TÂCHE 7/8
================================================================================
📋 Non-linéaire : tanh(3x) + 0.5x²
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 39.39 dB | MSE=0.000083                                                                                                                                                                  
SNR MLP          : 64.33 dB | MSE=0.000000
➡️  Gagnant : MLP (ΔSNR = -24.94 dB)

================================================================================
TÂCHE 8/8
================================================================================
📋 Sinusoïde bruitée (40 Hz, SNR=10dB)
   Données : 1000 échantillons

🔄 Réseau d'Oscillateurs Laplaciens
   Paramètres : 3521

🧠 MLP classique (tanh)                                                                                                                                                                                     
   Paramètres : 16897

SNR Oscillateurs : 39.56 dB | MSE=0.000061                                                                                                                                                                  
SNR MLP          : -0.01 dB | MSE=0.548692
➡️  Gagnant : Oscillateurs (ΔSNR = +39.57 dB)


================================================================================
RÉSUMÉ GLOBAL
================================================================================

Tâche                                       Osc SNR    MLP SNR      Δ SNR      Gagnant
Sinusoïde pure (50 Hz)                        65.94      -0.01      65.95 Oscillateurs
Chirp (10→200 Hz)                             30.70       0.25      30.45 Oscillateurs
AM (100 Hz ± 5 Hz)                            48.59      -0.02      48.62 Oscillateurs
Signal composite [20.0, 60.0, 150.0] Hz       36.49       0.01      36.48 Oscillateurs
Onde carrée (approx.)                         41.19       1.18      40.00 Oscillateurs
Onde en dents de scie (approx.)               31.99       0.51      31.47 Oscillateurs
Non-linéaire : tanh(3x) + 0.5x²               39.39      64.33     -24.94          MLP
Sinusoïde bruitée (40 Hz, SNR=10dB)           39.56      -0.01      39.57 Oscillateurs  