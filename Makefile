# Land on Earth — commandes. Tout passe par uv (https://docs.astral.sh/uv/).
# La clé OpenRouter est lue dans ton terminal (OPENROUTER_API_KEY), jamais dans un fichier du repo.
# `python -m loe` exécute directement le dossier loe/ du projet, sans passer par le lien que uv
# installe dans .venv (qu'iCloud peut rendre illisible).

UV := uv run --quiet
LOE := $(UV) python -m loe
GPT_OSS_PROVIDERS := coreweave,parasail,deepinfra,dekallm,akashml,cerebras
.PHONY: setup check pilot probe-gpt-oss run score status pack demo test truth clean-demo

setup:            ## installe Python et les dépendances
	uv sync

check: setup      ## gratuit : clé, plafond, fournisseurs, précisions, prix
	$(LOE) check

pilot: setup      ## quelques dizaines de centimes : stratégie de chaque modèle, 200 points, puis répétition générale aux réglages du run
	-$(LOE) check
	-$(LOE) pilot
	-$(LOE) preflight
	$(LOE) pack
	@printf "\nPousse maintenant results/ sur le repo (voir README, étape 4).\n"

probe-gpt-oss: setup ## < 1 centime : cherche un fournisseur qui fait tourner gpt-oss sans réflexion
	-$(LOE) probe --model gpt-oss-120b --provider $(GPT_OSS_PROVIDERS) --strategy raw
	-$(LOE) probe --model gpt-oss-20b --provider $(GPT_OSS_PROVIDERS),darkbloom --strategy raw
	$(LOE) pack

run: setup        ## ≈ 17 $, ~1 h 15 : 16 200 points × 24 modèles, seulement si la répétition générale est au vert ; relancer reprend
	$(LOE) run
	$(LOE) score
	$(LOE) pack
	@printf "\nPousse maintenant results/ sur le repo (voir README, étape 6).\n"

score:            ## recalcule classement et cartes à partir des réponses enregistrées
	$(LOE) score

status:           ## avancement et coût du run en cours (dans un autre terminal)
	$(LOE) status

pack:             ## compresse les réponses brutes pour le repo
	$(LOE) pack

demo: setup       ## tout le pipeline contre une fausse API, gratuit, dans results/demo/
	$(LOE) demo

test: setup       ## tests automatiques (fausse API)
	uv run --quiet pytest -q

truth:            ## reconstruit data/grid.csv (déjà dans le repo, inutile en temps normal)
	uv run --quiet --extra truth python scripts/build_truth.py

clean-demo:
	rm -rf results/demo
