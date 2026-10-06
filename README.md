# Land on Earth

**English** · [Français](#français)

Can a language model draw the world map from memory? Land on Earth asks 24 open-weight models “Land or Water?” at 16,200 coordinates, one every 2°, with no image and no tools. It redraws each model's map from its answers and scores it against a 1 km land mask, weighted by area. It reproduces the “Land or Water?” eval shared by Andrej Karpathy.

## English

### Results

MiniMax M3 (88.8 %) and DeepSeek V4.1 Flash (88.5 %) are tied at the top; DeepSeek V4.1 Flash got there for $0.10. Eight of the 24 models do no better than always answering “Water” (71.0 %), mostly because they answer “Land” far too often.

- **[RESULTS.md](RESULTS.md)**: the full analysis, the figures and how to read them.
- **[The full figure](results/tweet/figures/land_on_earth.png)**: what the models see together, where they go wrong, and every model's map.
- **[Hugging Face dataset](https://huggingface.co/datasets/solalzana/land-on-earth)**: every model's answer at every point, ready to load with `datasets`.

![Ranking of the 24 models](results/tweet/figures/ranking.png)

Run of 6 October 2026: 389,196 requests, $16.85, about an hour and a half.

### How it works

**The question**, the same for every model:

```text
Coordinates: 41.0, -73.0
Land or Water? Answer with one word: Land or Water.
```

**The grid and the truth.** 16,200 points, the centres of 2° cells (latitudes −89 to 89, longitudes −179 to 179), compared with the GLOBE 1 km land mask (`global-land-mask`), which is 29.0 % land by area. The score is the share of the globe's area answered correctly, each point weighted by the cosine of its latitude; always answering “Water” already scores 71.0 %. A variant counts large lakes and the Caspian Sea as water (48 points change).

**One provider per model.** Every request goes through [OpenRouter](https://openrouter.ai), pinned to a single provider at a known precision. There is no silent fallback to another provider (often in FP4), and any provider that would ignore a parameter is refused. The provider that actually answered is stored with each answer.

```json
"provider": {"only": ["parasail"], "allow_fallbacks": false, "require_parameters": true, "quantizations": ["fp8"]}
```

**Seven ways to ask.** A pilot tries them in order and keeps the first that works: without reasoning first, with logprobs then without, and minimal reasoning only as a last resort.

| Strategy | When |
|---|---|
| `chat_none` | chat, reasoning switched off (`reasoning: {effort: none}`): the normal case |
| `chat` | models without a reasoning parameter (Llama) |
| `raw` | raw prompt: the model's official chat template (Hugging Face) with the answer prefilled, when the provider passes it through unchanged |
| `text_none`, `text` | like `chat_none` and `chat`, without logprobs, for providers that do not return them (Mistral, SiliconFlow) or refuse the parameter |
| `chat_low`, `text_low` | last resort, for models that cannot answer without reasoning (gpt-oss, GLM-5.3, MiniMax M3): minimal reasoning, with or without logprobs |

A strategy passes if at least 7 of 8 answers arrive, are readable, show no reasoning (except `chat_low` and `text_low`) and come from the pinned provider. The temperature is 0 everywhere except for Kimi K3 at Alibaba, which does not accept the parameter; its answer is read from the logprobs, so it is still the more probable of the two words.

**Reading the answer.** At the first token that is not markup, P(Land) = mass of “Land” / (mass of “Land” + mass of “Water”), adding up the variants (` Land`, `land`, `LAND`…) among the alternatives the provider returns. The prediction is the more probable of the two. Without logprobs, the first “Land” or “Water” in the text is the answer.

**Safeguards.** Before the run, a dress rehearsal asks 300 new points per model with exactly the run's settings and gives a green or red light, with the duration, the cost and who pays what. During the run:

- a global budget, $22 by default, counting what provider keys bill directly;
- a model stops if its projected cost goes above 2.5 times its estimate or 1.5 times the pilot's projection, or after 25 errors in a row (rate limits aside);
- rate limits and server errors are retried with exponential backoff;
- a refused OpenRouter key or empty credits stop everything.

An interrupted run resumes where it stopped and only asks again the points that failed or came back empty.

**Pace.** Each model has its number of simultaneous requests (8 to 40) and, when its provider has a limit, a pace just below it (`rpm`), so requests are spaced out instead of being refused and retried.

### Reproduce

You need macOS or Linux, [uv](https://docs.astral.sh/uv/) and an OpenRouter API key. A full run costs about $17. With the provider keys below it takes about an hour and a half. The command-line messages are in French.

**1. Install**

```sh
git clone https://github.com/solalz1/land-on-earth-benchmark.git
cd land-on-earth-benchmark
make setup
```

On macOS, keep the clone out of folders synced with iCloud (Desktop, Documents): iCloud damages the Python environment and slows the run down.

**2. Set the OpenRouter key.** The code reads it from the environment and never prints it or writes it anywhere. A spending limit on the key (openrouter.ai/settings/keys) is a good safeguard.

```sh
export OPENROUTER_API_KEY=sk-or-...   # for instance in ~/.zshrc
```

**3. Provider keys (optional, for speed).** On new accounts, OpenRouter allows 20 requests per minute per model on 8 of these models: about 14 hours for one map. With your own key at the provider, your account's limits there apply instead, and the provider bills you directly. Add each key in OpenRouter (settings > integrations), under **Prioritized** with **Use shared capacity**, filtered on its models:

| Key | Models (filter) | Limit |
|---|---|---|
| SiliconFlow (prepaid: keep about $7 of balance) | `deepseek/deepseek-v4-pro`, `moonshotai/kimi-k2.6`, `z-ai/glm-5.2`, `z-ai/glm-5.3` | 500 requests per minute per model |
| Alibaba Cloud (Model Studio, Singapore region) | `moonshotai/kimi-k3`, `qwen/qwen3.5-397b-a17b`, `qwen/qwen3.5-122b-a10b`, `qwen/qwen3.6-27b` | 600 requests per minute per Qwen model |
| Mistral | no filter: the 5 Mistral models | per model, see admin.mistral.ai > Limits |

The other 11 models use OpenRouter credits. The dress rehearsal checks that each key serves its models, and only those. Without these keys, set `byok: false` for those models in `configs/models.yaml` and lower their `rpm` below OpenRouter's limit.

**4. Check, for free**

```sh
make demo     # the whole pipeline against a fake API, in results/demo/
make check    # key and limit, each model's provider, precision, logprobs and actual prices
```

**5. Pilot and dress rehearsal** (a few tens of cents, about 15 minutes). The pilot finds how to query each model on 8 points, asks 200 points of each and projects the cost of the run; then comes the dress rehearsal. Reports: `results/pilot/report.md` and `results/pilot/preflight.md`.

```sh
make pilot
```

**6. Full run** (about $17). It only starts if the dress rehearsal is green and `configs/models.yaml` has not changed since. If it stops, or ends with “Run incomplet”, run the same command again to resume. `make status`, in another terminal, shows the progress and the cost.

```sh
make run      # on macOS, caffeinate -i make run keeps the Mac awake
```

### Commands

| Command | Cost | What it does |
|---|---|---|
| `make check` | free | reads the key (limit, balance) and each model's OpenRouter listing |
| `make demo` | free | check, pilot, run, leaderboard and maps against a fake API |
| `make pilot` | a few tens of cents | each model's strategy, 200 points, report, then the dress rehearsal (300 points per model at the run's settings): green or red, cost, duration and who pays |
| `make probe-gpt-oss` | under one cent | looks for a provider that runs gpt-oss without reasoning |
| `make run` | about $17 | 16,200 points × the models validated by the pilot, then leaderboard, maps, figures, compression |
| `make status` | free | progress and cost of the run |
| `make score` | free | leaderboard and maps again, from the stored answers |
| `make figure` | free | the figures and numbers of [RESULTS.md](RESULTS.md) (`results/tweet/figures/`) |
| `make hf` | free | `hf/`, the Hugging Face dataset, to send with `hf upload <user>/<dataset> hf --repo-type=dataset` |
| `make test` | free | automated tests, against the fake API |

Useful options: `uv run python -m loe run --models qwen3.5-9b,gemma-4-31b` for a few models; `--budget 10` to cap the cost of the run, credits and provider keys included ($22 by default). `uv run python -m loe preflight` runs the dress rehearsal alone. `uv run python -m loe probe --model gpt-oss-20b --provider deepinfra,parasail --strategy raw` tries a model at providers other than the configured one, 8 points per try.

### Files

```text
configs/models.yaml     the 24 models: OpenRouter id, provider, precision, price, strategies, pace, key
data/grid.csv           the grid and the truth (make truth rebuilds it)
loe/                    the code: OpenRouter client, pilot, run, scoring, maps, figures, fake API
scripts/build_truth.py  builds data/grid.csv
tests/                  tests against the fake API
results/pilot/          pilot report, chosen strategies, dress rehearsal, compressed raw answers
results/probe/          tries at other providers (make probe-gpt-oss)
results/tweet/          the full run: raw answers (raw/*.jsonl.gz), predictions, leaderboard, maps, figures
RESULTS.md              the results and how to read them
```

### License

The code is under the MIT license. The dataset on Hugging Face is under CC BY 4.0. By Solal Zana.

## Français

Un modèle de langage sait-il dessiner la carte du monde de mémoire ? Land on Earth pose « Land or Water? » à 24 modèles open-weight sur 16 200 coordonnées, une tous les 2°, sans image ni outil. Il redessine la carte de chaque modèle à partir de ses réponses et la note contre un masque terre à 1 km, pondéré par la surface. C'est une reproduction de l'éval « Land or Water? » relayée par Andrej Karpathy.

### Résultats

MiniMax M3 (88,8 %) et DeepSeek V4.1 Flash (88,5 %) sont à égalité en tête ; DeepSeek V4.1 Flash y arrive pour 0,10 $. Huit modèles sur 24 ne font pas mieux que répondre toujours « Water » (71,0 %), surtout parce qu'ils répondent « Land » bien trop souvent.

- **[RESULTS.md](RESULTS.md)** : l'analyse complète, les figures et comment les lire (en anglais).
- **[La figure complète](results/tweet/figures/land_on_earth.png)** : ce que voient les modèles ensemble, où ils se trompent, et la carte de chaque modèle.
- **[Le dataset sur Hugging Face](https://huggingface.co/datasets/solalzana/land-on-earth)** : la réponse de chaque modèle en chaque point, prête à charger avec `datasets`.

Le classement est la figure de la version anglaise, plus haut. Run du 6 octobre 2026 : 389 196 requêtes, 16,85 $, environ une heure et demie.

### Comment ça marche

**La question**, identique pour tous les modèles :

```text
Coordinates: 41.0, -73.0
Land or Water? Answer with one word: Land or Water.
```

**La grille et la vérité terrain.** 16 200 points, les centres des cellules de 2° (latitudes −89 à 89, longitudes −179 à 179), comparés au masque terre GLOBE à 1 km (`global-land-mask`), qui compte 29,0 % de terre en surface. Le score est la part de la surface du globe correctement classée, chaque point pondéré par le cosinus de sa latitude ; répondre toujours « Water » fait déjà 71,0 %. Une variante compte les grands lacs et la mer Caspienne comme de l'eau (48 points changent).

**Un fournisseur par modèle.** Chaque requête passe par [OpenRouter](https://openrouter.ai), épinglée chez un seul fournisseur, à une précision connue. Il n'y a pas de repli silencieux vers un autre fournisseur (souvent en FP4), et tout fournisseur qui ignorerait un paramètre est refusé. Le fournisseur qui a réellement répondu est enregistré avec chaque réponse.

**Sept façons de poser la question** (tableau de la version anglaise). Un pilote les essaie dans l'ordre et garde la première qui marche : sans réflexion d'abord, avec logprobs puis sans, et la réflexion minimale seulement en dernier recours, pour les modèles qui ne peuvent pas répondre sans réfléchir (gpt-oss, GLM-5.3, MiniMax M3). Une stratégie passe si au moins 7 réponses sur 8 arrivent, sont lisibles, ne montrent pas de réflexion (sauf `chat_low` et `text_low`) et viennent du fournisseur épinglé. La température est à 0 partout, sauf pour Kimi K3 chez Alibaba, qui n'accepte pas le paramètre ; sa réponse est lue dans les logprobs, donc reste la plus probable des deux.

**La lecture de la réponse.** Au premier token qui n'est pas de la mise en forme, P(Land) = masse de « Land » / (masse de « Land » + masse de « Water »), en additionnant les variantes (` Land`, `land`, `LAND`…) parmi les alternatives renvoyées par le fournisseur. La prédiction est la plus probable des deux. Sans logprobs, la réponse est le premier « Land » ou « Water » du texte.

**Les garde-fous.** Avant le run, une répétition générale pose 300 nouveaux points par modèle, avec exactement les réglages du run, et donne un feu vert ou rouge, avec la durée, le coût et qui paie quoi. Pendant le run :

- un budget global, 22 $ par défaut, qui compte aussi ce que facturent les clés fournisseurs ;
- un modèle s'arrête si son coût projeté dépasse 2,5 fois son estimation ou 1,5 fois la projection du pilote, ou après 25 erreurs d'affilée (hors limites de débit) ;
- les limites de débit et les erreurs serveur sont réessayées avec un backoff exponentiel ;
- une clé OpenRouter refusée ou des crédits épuisés arrêtent tout.

Un run interrompu reprend là où il s'est arrêté et ne repose que les points en erreur ou revenus vides.

**Le rythme.** Chaque modèle a son nombre de requêtes simultanées (8 à 40) et, quand son fournisseur a une limite, un rythme juste en dessous (`rpm`) : les requêtes sont espacées au lieu d'être refusées puis réessayées.

### Reproduire

Il faut macOS ou Linux, [uv](https://docs.astral.sh/uv/) et une clé API OpenRouter. Un run complet coûte environ 17 $. Avec les clés fournisseurs ci-dessous, il dure environ une heure et demie.

**1. Installer**

```sh
git clone https://github.com/solalz1/land-on-earth-benchmark.git
cd land-on-earth-benchmark
make setup
```

Sur macOS, mieux vaut garder le dossier hors des dossiers synchronisés par iCloud (Bureau, Documents) : iCloud abîme l'environnement Python et ralentit le run.

**2. La clé OpenRouter.** Le code la lit dans l'environnement, ne l'affiche jamais et ne l'écrit nulle part. Un plafond de dépense sur la clé (openrouter.ai/settings/keys) est une bonne précaution.

```sh
export OPENROUTER_API_KEY=sk-or-...   # par exemple dans ~/.zshrc
```

**3. Les clés fournisseurs (facultatif, pour aller vite).** Sur les comptes récents, OpenRouter limite 8 de ces modèles à 20 requêtes par minute chacun : environ 14 h pour une carte. Avec sa propre clé chez le fournisseur, ce sont les limites de ce compte qui s'appliquent, et le fournisseur facture directement. Chaque clé s'ajoute dans OpenRouter (settings > integrations), en **Prioritized** avec **Use shared capacity**, filtrée sur ses modèles : SiliconFlow pour DeepSeek V4 Pro, Kimi K2.6, GLM-5.2 et GLM-5.3 (prépayé : prévoir environ 7 $ de solde) ; Alibaba Cloud (Model Studio, région Singapour) pour Kimi K3, Qwen3.5-397B, Qwen3.5-122B et Qwen3.6-27B ; Mistral, sans filtre, pour les 5 modèles Mistral. Les identifiants exacts sont dans le tableau de la version anglaise. Les 11 autres modèles passent par les crédits OpenRouter. La répétition générale vérifie que chaque clé sert bien ses modèles, et seulement eux. Sans ces clés, il faut mettre `byok: false` pour ces modèles dans `configs/models.yaml` et baisser leur `rpm` sous la limite d'OpenRouter.

**4. Vérifier, gratuitement**

```sh
make demo     # tout le pipeline contre une fausse API, dans results/demo/
make check    # clé et plafond, fournisseur de chaque modèle, précision, logprobs et prix réels
```

**5. Pilote et répétition générale** (quelques dizaines de centimes, environ un quart d'heure). Le pilote cherche comment interroger chaque modèle sur 8 points, en pose 200 à chacun et projette le coût du run ; vient ensuite la répétition générale. Rapports : `results/pilot/report.md` et `results/pilot/preflight.md`.

```sh
make pilot
```

**6. Run complet** (environ 17 $). Il ne part que si la répétition générale est au vert et que `configs/models.yaml` n'a pas changé depuis. S'il s'arrête, ou finit par « Run incomplet », la même commande le fait reprendre. `make status`, dans un autre terminal, affiche l'avancement et le coût.

```sh
make run      # sur macOS, caffeinate -i make run garde le Mac éveillé
```

### Commandes

| Commande | Coût | Ce qu'elle fait |
|---|---|---|
| `make check` | gratuit | lit la clé (plafond, reste) et la fiche OpenRouter de chaque modèle |
| `make demo` | gratuit | check, pilote, run, classement et cartes contre une fausse API |
| `make pilot` | quelques dizaines de centimes | stratégie de chaque modèle, 200 points, rapport, puis répétition générale (300 points par modèle aux réglages du run) : feu vert ou rouge, coût, durée et qui paie |
| `make probe-gpt-oss` | moins d'un centime | cherche un fournisseur qui fait tourner gpt-oss sans réflexion |
| `make run` | environ 17 $ | 16 200 points × les modèles validés au pilote, puis classement, cartes, figures, compression |
| `make status` | gratuit | avancement et coût du run |
| `make score` | gratuit | recalcule classement et cartes à partir des réponses enregistrées |
| `make figure` | gratuit | les figures et les chiffres de [RESULTS.md](RESULTS.md) (`results/tweet/figures/`) |
| `make hf` | gratuit | `hf/`, le dataset Hugging Face, à envoyer avec `hf upload <utilisateur>/<dataset> hf --repo-type=dataset` |
| `make test` | gratuit | tests automatiques, contre la fausse API |

Options utiles : `uv run python -m loe run --models qwen3.5-9b,gemma-4-31b` pour quelques modèles ; `--budget 10` pour plafonner le coût du run, crédits et clés fournisseurs compris (22 $ par défaut). `uv run python -m loe preflight` relance la répétition générale seule. `uv run python -m loe probe --model gpt-oss-20b --provider deepinfra,parasail --strategy raw` essaie un modèle chez d'autres fournisseurs que celui de la config, 8 points par essai.

### Fichiers

```text
configs/models.yaml     les 24 modèles : identifiant OpenRouter, fournisseur, précision, prix, stratégies, rythme, clé
data/grid.csv           la grille et la vérité terrain (make truth la reconstruit)
loe/                    le code : client OpenRouter, pilote, run, scoring, cartes, figures, fausse API
scripts/build_truth.py  construction de data/grid.csv
tests/                  tests contre la fausse API
results/pilot/          rapport du pilote, stratégies retenues, répétition générale, réponses brutes compressées
results/probe/          essais chez d'autres fournisseurs (make probe-gpt-oss)
results/tweet/          le run complet : réponses brutes (raw/*.jsonl.gz), prédictions, classement, cartes, figures
RESULTS.md              les résultats et comment les lire
```

### Licence

Le code est sous licence MIT. Le dataset sur Hugging Face est sous licence CC BY 4.0. Par Solal Zana.
