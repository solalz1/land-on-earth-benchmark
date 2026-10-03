# Land on Earth

Reproduction de l'éval « Land or Water? » relayée par Karpathy, sur 20 modèles open-weight. On demande au modèle « terre ou eau ? » pour 16 200 coordonnées, une tous les 2°, sans image ni outil, puis on redessine sa carte du monde et on la note contre un masque terre à 1 km, pondéré par la surface.

![Vérité terrain](docs/truth.png)

Coût prévu : environ 12 € de crédits OpenRouter pour les 20 modèles (324 000 requêtes), plus environ 0,15 € pour le pilote.

## Ce que tu fais, dans l'ordre

Tout se lance depuis le Terminal de ton Mac, dans le dossier du repo.

**1. Installer uv** (gère Python et les dépendances), une seule fois :

```sh
brew install uv          # ou : curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/solalz1/land-on-earth-benchmark.git
cd land-on-earth-benchmark
make setup
```

**2. Ajouter la clé OpenRouter à ton terminal**, une seule fois. Ouvre `~/.zshrc` dans un éditeur (`open -e ~/.zshrc`), ajoute la ligne ci-dessous avec ta clé, enregistre, puis ouvre un nouveau terminal.

```sh
export OPENROUTER_API_KEY=sk-or-...
```

> Ne colle jamais la clé dans le chat, dans un fichier du repo ou dans une commande `git`. Le code la lit dans l'environnement, ne l'affiche jamais et ne l'écrit nulle part. Plafonne-la à 20 $ sur openrouter.ai/settings/keys.

**3. Vérifier, gratuitement** :

```sh
make demo     # facultatif : tout le pipeline contre une fausse API, dans results/demo/
make check    # clé et plafond, fournisseur de chaque modèle, précision, logprobs, prix réels
```

**4. Pilote (≈ 0,15 €, quelques minutes)** : choisit la bonne façon d'interroger chaque modèle sur 8 points, puis pose 200 points à chacun et projette le coût du run.

```sh
make pilot
git add results && git commit -m "Pilote" && git push
```

Le rapport est dans `results/pilot/report.md`. Claude le relit et corrige la config si un modèle échoue.

**5. Run complet (≈ 12 €, de 30 min à 4 h)**, Mac éveillé :

```sh
caffeinate -i make run
```

S'il s'arrête (coupure réseau, Mac en veille, Ctrl-C), relance la même commande : il reprend là où il s'est arrêté, sans reposer les questions déjà répondues. Dans un autre terminal, `make status` affiche l'avancement et le coût.

**6. Pousser les résultats** :

```sh
git add results && git commit -m "Run complet" && git push
```

## Commandes

| Commande | Coût | Ce qu'elle fait |
|---|---|---|
| `make check` | gratuit | lit la clé (plafond, reste) et la fiche OpenRouter de chaque modèle |
| `make demo` | gratuit | check, pilote, run, classement et cartes contre une fausse API |
| `make pilot` | ≈ 0,15 € | stratégie de chaque modèle, 200 points, rapport et projection de coût |
| `make run` | ≈ 12 € | 16 200 points × modèles validés au pilote, puis classement, cartes, compression |
| `make status` | gratuit | avancement et coût du run |
| `make score` | gratuit | recalcule classement et cartes depuis les réponses enregistrées |
| `make test` | gratuit | tests automatiques, contre la fausse API |

Options utiles : `uv run loe run --models qwen3.5-9b,gemma-4-31b` pour quelques modèles, `--budget 10` pour plafonner les crédits du run (18 $ par défaut).

## Comment ça marche

**La question**, identique pour les 20 modèles, à température 0 :

```text
Coordinates: 41.0, -73.0
Land or Water? Answer with one word: Land or Water.
```

**Le fournisseur est épinglé** à chaque requête : un seul fournisseur autorisé, pas de repli silencieux vers un autre (souvent en FP4), et refus de tout fournisseur qui ignorerait un paramètre. Le fournisseur qui a réellement répondu est enregistré avec chaque réponse.

```json
"provider": {"only": ["parasail"], "allow_fallbacks": false, "require_parameters": true, "quantizations": ["fp8"]}
```

**Cinq façons de poser la question**, essayées dans l'ordre par le pilote, qui garde la première qui marche :

| Stratégie | Quand |
|---|---|
| `chat_none` | chat, réflexion coupée (`reasoning: {effort: none}`) : le cas normal |
| `chat` | modèles sans paramètre de réflexion (Llama) |
| `raw` | prompt brut : template officiel du modèle (Hugging Face) + réponse pré-remplie, pour les modèles qui raisonnent malgré la consigne (gpt-oss, Kimi K3) |
| `chat_low` | dernier recours : réflexion minimale, réponse lue après |
| `text` | fournisseur sans logprobs (Mistral) : réponse à température 0, comme la carte du tweet |

Une stratégie passe si au moins 7 réponses sur 8 arrivent, sont lisibles, contiennent Land et Water dans les 20 meilleurs logprobs, ne montrent pas de réflexion, et viennent du fournisseur épinglé.

**La réponse** : au premier token qui n'est pas de la mise en forme, P(Land) = masse de « Land » / (masse de « Land » + masse de « Water »), en additionnant les variantes (` Land`, `land`, `LAND`…) parmi les 20 meilleurs. La prédiction est la plus probable des deux. Sans logprobs, on lit le premier « Land » ou « Water » du texte.

**Les garde-fous du run** : budget global (18 $ par défaut, compté sur les réponses déjà enregistrées) ; arrêt d'un modèle dont le coût projeté dépasse 2,5 fois son estimation (un modèle qui se met à raisonner) ; arrêt d'un modèle après 25 erreurs d'affilée ; arrêt immédiat si la clé est refusée ou les crédits épuisés ; reprise avec backoff exponentiel sur les erreurs 429 et 5xx.

**Les scores** (`results/tweet/leaderboard.md`) :

- **Précision** : part de la surface du globe correctement classée (poids = cosinus de la latitude) ; un point sans réponse compte faux.
- **Skill** : gain sur la réponse constante « toujours Water », qui fait déjà 71,0 % : (précision − 0,71) / (1 − 0,71).
- **Avec lacs** : même précision, en comptant comme eau les grands lacs et la Caspienne, que le masque 1 km classe en terre.
- Rappel et précision sur la terre, couverture, score de Brier quand il y a des logprobs.

**Les cartes** (`results/tweet/maps/`) : une par modèle (blanc = terre, noir = eau, gris = sans réponse), la carte de probabilité (`prob/`), la carte des erreurs (`errors/` : rouge = terre répondue sur de l'eau, bleu = eau répondue sur de la terre), et `montage.png`, la figure au format du tweet.

## Vérité terrain

`data/grid.csv` contient les 16 200 points (centres des cellules de 2°, latitudes −89 à 89, longitudes −179 à 179) avec :

- `truth` : masque terre GLOBE à 1 km (paquet `global-land-mask`), au plus près du « 1-km land mask » du tweet ; 29,0 % de terre en surface.
- `truth_lakes` : le même, avec les lacs Natural Earth 1:10m et la mer Caspienne comptés comme eau (48 points changent).
- `weight` : cosinus de la latitude, proportionnel à la surface de la cellule.

Le fichier est commité ; `make truth` le reconstruit.

## Fichiers

```text
configs/models.yaml     les 20 modèles : identifiant OpenRouter, fournisseur, précision, prix, stratégies
data/grid.csv           la grille et la vérité terrain
loe/                    le code : client OpenRouter, pilote, run, scoring, cartes, fausse API
scripts/build_truth.py  construction de data/grid.csv
tests/                  tests contre la fausse API
results/check.json      sortie de make check
results/pilot/          rapport du pilote, stratégies retenues, réponses brutes compressées
results/tweet/          run complet : réponses (raw/*.jsonl.gz), prédictions, classement, cartes
```
