# Abstrait

Je présente une nouvelle architecture neuronale qui repense fondamentalement la façon dont nous abordons l'apprentissage des signaux temporels et le contrôle robotique. Le **Laplace Perceptron** exploite la décomposition spectro-temporelle avec des harmoniques amorties à valeurs complexes, offrant à la fois une représentation supérieure du signal analogique et un chemin à travers des espaces de solutions complexes qui permettent d'échapper aux minima locaux dans les paysages d'optimisation.

# Pourquoi c'est important

Les réseaux neuronaux traditionnels discrétisent le temps et traitent les signaux comme des séquences d'échantillons indépendants. Cela fonctionne, mais cela ne correspond pas fondamentalement à la manière dont les systèmes physiques (robots, audio, dessins) fonctionnent réellement en temps continu. Le Perceptron de Laplace modélise les signaux sous forme d'**oscillateurs harmoniques amortis dans le domaine fréquentiel**, à l'aide de paramètres apprenables qui ont des interprétations physiques directes.

Plus important encore, en opérant dans le **domaine complexe** (grâce à des bases sinus/cosinus couplées avec phase et amortissement), le paysage d'optimisation devient plus riche. Les représentations à valeurs complexes permettent à la descente de gradient d'explorer des variétés de solutions inaccessibles aux réseaux à valeurs purement réelles, offrant potentiellement des voies de sortie des minima locaux qui piègent les architectures traditionnelles.

# Architecture de base

L’élément fondamental combine :

1. **Bases spectro-temporelles** : Chaque unité génère un oscillateur amorti : y\_k(t) = exp(-s\_k \* t) \* \[a\_k \* sin(ω\_k \* t + φ\_k) + b\_k \* cos(ω\_k \* t + φ\_k)\]
2. **Espace de paramètres complexe** : le couplage entre les composants sinus/cosinus avec des phases apprenables crée une représentation à valeurs complexes où l'optimisation peut exploiter à la fois les gradients d'amplitude et de phase.
3. **Interprétabilité physique** :
   * `s_k`: coefficient d'amortissement (taux de décroissance)
   * `ω_k`: fréquence angulaire
   * `φ_k`: déphasage
   * `a_k, b_k`: composantes d'amplitude complexes

# Pourquoi les solutions complexes aident à échapper aux minimums locaux

C'est la percée théorique : lors de l'optimisation dans un espace complexe, le paysage des pertes a des propriétés topologiques différentes de celles de sa projection en valeur réelle. Spécifiquement:

* **Structure de gradient plus riche** : les gradients complexes fournissent des informations en deux dimensions (réelle/imaginaire ou amplitude/phase) plutôt qu'une seule.
* **Diversité de phases** : plusieurs solutions peuvent partager des magnitudes similaires mais différer en phase, créant des chemins continus entre les optima locaux
* **Convexité dans le domaine fréquentiel** : certains problèmes non convexes dans le domaine temporel se comportent mieux dans l'espace fréquentiel
* **Régularisation naturelle** : le couplage entre les termes sinus/cosinus crée des contraintes implicites qui peuvent lisser le paysage d'optimisation

Pensez-y comme ceci : si votre surface d'erreur a une vallée (minimum local), les gradients traditionnels à valeur réelle ne peuvent grimper que le long d'un seul axe. L'optimisation à valeurs complexes peut « s'exprimer » en ajustant simultanément l'ampleur et la phase, accédant ainsi à des trajectoires d'évasion qui n'existent pas dans l'espace purement réel.

# Portefeuille de mise en œuvre

J'ai développé cinq implémentations démontrant la polyvalence de cette architecture :

# 1. Contrôle robotique inter-espace ([`12-laplace_jointspace_fk.py`](https://github.com/stakepoolplace/laplace-perceptron))

Cette implémentation contrôle un **bras robotique à 6 DOF** à l'aide de la cinématique avant. Au lieu d'apprendre la cinématique inverse (difficile !), il paramétrise les angles articulaires θ\_j(t) comme des sommes d'harmoniques de Laplace :

    classe LaplaceJointEncoder(nn.Module) :
        def forward(soi, t_grid) :
            décroissance = torch.exp(-s * t)
            sinwt = torch.sin(w * t)
            coswt = torche.cos(w * t)
            série = désintégration * (a * sinwt + b * coswt)
            thêta = série.sum(dim=-1) + thêta0
            retourner thêta

**Résultat clé** : Apprend des trajectoires douces et naturelles (cercles, lemniscates) à travers l'espace articulaire en optimisant seulement \~400 paramètres. La représentation harmonique complexe encourage naturellement des mouvements physiquement réalisables avec des profils d'accélération continus.

Le code comprend de superbes visualisations 3D montrant le bras traçant les trajectoires cibles avec un rapport hauteur/largeur de 1:1:1 et une rotation de la caméra en option.

# 2. Apprentissage temporel synchronisé ([`6-spectro-laplace-perceptron.py`](https://github.com/stakepoolplace/laplace-perceptron))

Démontre la **synchronisation Kuramoto** entre les unités d'oscillateurs : un phénomène issu de la physique dans lequel les oscillateurs couplés se verrouillent naturellement en phase. Cela crée une coordination temporelle émergente :

    phase_mean = osc_phase.mean(dim=2)
    diff = phase_mean.unsqueeze(2) - phase_mean.unsqueeze(1)
    sync_term = torch.sin(diff).mean(dim=2)
    phi_new = phi_prev + K_phase * sync_term

Le modèle apprend à représenter des signaux multifréquences complexes (sommes amorties de sinus/cosinus) tout en maintenant la cohérence de phase entre les unités. Les courbes de perte montrent une convergence stable même pour des cibles hautement non stationnaires.

# 3. Apprentissage spectral audio ([`7-spectro_laplace_audio.py`](https://github.com/stakepoolplace/laplace-perceptron))

Applique l'architecture à **la synthèse de forme d'onde audio**. En paramétrant le son sous forme de séries harmoniques amorties, il capture naturellement :

* Structure formant (fréquences de résonance)
* Dégradation temporelle (attaques/libérations d'instruments)
* Relations harmoniques (intervalles musicaux)

La représentation complexe est particulièrement puissante ici car la perception audio est intrinsèquement un domaine fréquentiel et les relations de phase déterminent le timbre.

# 4. Contrôle de dessin continu ([`8-laplace_drawing_face.py`](https://github.com/stakepoolplace/8-laplace_drawing_face.py))

Peut-être la démo la plus convaincante visuellement : apprendre à dessiner des dessins au trait continu (par exemple, des visages) en représentant les trajectoires du stylo x(t), y(t) sous la forme d'une série de Laplace. Le réseau apprend :

* Traits fluides et naturels (l'amortissement empêche le tremblement)
* Séquençage approprié (relations de phases)
* Profils de pression/vitesse implicitement

C'est vraiment difficile pour les RNN/Transformers car ils discrétisent le temps. L'approche Laplace traite le dessin comme ce qu'il est physiquement : un mouvement continu.

# 5. Transformateur-Laplace Hybrid ([`13-laplace-transformer.py`](https://github.com/stakepoolplace/13-laplace-transformer.py))

Intègre les perceptrons de Laplace en tant que **codages de position continus** dans les architectures de transformateur. Au lieu d'intégrations sinusoïdales fixes, il utilise des harmoniques amorties apprenables :

    pos_encoding = laplace_encoder(time_grid) # [T, d_model]
    x = x + pos_encodage

Cela permet aux transformateurs de :

* Apprendre les échelles temporelles spécifiques aux tâches
* Adapter la douceur de l'encodage via l'amortissement
* Représente des modèles apériodiques/transitoires

Les premières expériences montrent des performances améliorées en matière de prévision de séries chronologiques par rapport aux codages positionnels standard.

# Pourquoi cette architecture excelle en robotique

Plusieurs propriétés rendent les perceptrons de Laplace idéaux pour le contrôle robotique :

1. **Garanties de continuité** : Les harmoniques amorties sont infiniment différenciables → vitesses/accélérations douces
2. **Paramétrage physique** : l'amortissement/la fréquence ont des interprétations directes comme une dynamique naturelle
3. **Représentation efficace** : peu de paramètres (10 à 100 harmoniques) capturent des trajectoires complexes
4. **Extrapolation** : l'apprentissage dans le domaine fréquentiel se généralise mieux temporellement que les RNN
5. **Efficacité du calcul** : pas de récurrence → parallélisable, pas de gradients qui disparaissent

L'aspect à valeur complexe aide spécifiquement à **l'optimisation de trajectoire**, où nous devons échapper aux minima locaux correspondant aux configurations conjointes qui entrent en collision ou violent les contraintes de l'espace de travail. La descente de pente traditionnelle reste bloquée ; une optimisation complexe peut contourner ces obstacles en explorant l’espace des phases.

# Implications théoriques

Ce travail relie plusieurs idées profondes :

* **Traitement du signal** : Théorie des systèmes linéaires, transformées de Laplace, analyse harmonique
* **Systèmes dynamiques** : Réseaux d'oscillateurs, phénomènes de synchronisation
* **Analyse complexe** : fonctions holomorphes, surfaces de Riemann, optimisation complexe
* **Contrôle moteur** : générateurs de motifs centraux, synergies musculaires, trajectoires à jerk minimum

Le fait qu'une architecture unique unifie ces domaines suggère que nous avons trouvé quelque chose de fondamental sur la manière dont les systèmes continus doivent être appris.

# Questions ouvertes et travaux futurs

1. **Garanties théoriques** : Pouvons-nous prouver des taux de convergence ou des conditions d'optimalité pour une optimisation à valeurs complexes dans ce contexte ?
2. **Stabilité** : Comment pouvons-nous garantir que la dynamique apprise reste stable (tous les pôles dans le demi-plan gauche) ?
3. **Évolutivité** : cette approche fonctionne-t-elle pour les systèmes à plus de 100 DOF (humanoïdes) ?
4. **Architectures hybrides** : Comment combiner au mieux avec le raisonnement discret (transformateurs, RL) ?
5. **Plausibilité biologique** : Les neurones corticaux mettent-ils en œuvre quelque chose comme ça pour le contrôle moteur ?

# Conclusion

Le Perceptron de Laplace représente un changement de paradigme : au lieu de forcer des signaux continus dans des architectures neuronales discrètes, nous construisons des réseaux qui fonctionnent nativement en temps continu avec des représentations à valeurs complexes. Ce n'est pas seulement plus propre mathématiquement : cela change fondamentalement le paysage de l'optimisation, offrant des chemins à travers des espaces de solutions complexes qui aident à échapper aux minimums locaux.

Pour la robotique et l’apprentissage du mouvement en particulier, cela signifie que nous pouvons apprendre des comportements plus fluides, plus naturels et plus généralisables avec moins de paramètres et une meilleure efficacité des échantillons. Les cinq implémentations que j'ai partagées le démontrent dans les architectures de dessin, audio, de manipulation et hybrides.

**L'idée clé** : en adoptant le domaine complexe, nous ne nous contentons pas de mieux représenter les signaux : nous modifions la géométrie de l'apprentissage lui-même.

# Disponibilité des codes

Les cinq implémentations avec une documentation complète, des outils de visualisation et des exemples entraînés : [Dépôt GitHub](#) *(remplacer par le lien réel)*

Chaque fichier est autonome avec des commentaires détaillés et peut être exécuté avec :

    python 12-laplace_jointspace_fk.py --trajectory circle_wavy --n_units 100 --epochs 2000 --n_points 200 

# Références

*Articles clés qui ont inspiré ce travail :*

* Laplace transforme les réseaux de neurones (littérature récente sur l'apprentissage profond)
* Modèles Kuramoto et théorie de la synchronisation
* Réseaux de neurones à valeurs complexes (Hirose, Nitta)
* Primitives motrices et optimisation de trajectoire
* Méthodes spectrales en deep learning

**TL;DR** : J'ai construit un nouveau type de perceptron qui représente les signaux sous forme d'harmoniques amorties dans le domaine complexe. Il est meilleur pour apprendre les mouvements continus (robots, dessin, audio) car il fonctionne avec la structure de fréquence naturelle de ces signaux. Plus important encore, opérer dans un espace complexe permet à l’optimisation d’échapper aux minima locaux en fournissant des informations de gradient plus riches. Cinq implémentations fonctionnelles incluses pour les architectures robotiques, audio et hybrides.

*Qu'en penses-tu? Quelqu'un d'autre a-t-il exploré la décomposition temporelle à valeurs complexes pour l'apprentissage du mouvement ? J'aimerais entendre des commentaires sur la théorie et les applications pratiques.*
