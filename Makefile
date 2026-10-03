# Land on Earth — commandes. Tout passe par uv (https://docs.astral.sh/uv/).
# La clé OpenRouter est lue dans ton terminal (OPENROUTER_API_KEY), jamais dans un fichier du repo.

UV := uv run --quiet
.PHONY: setup check pilot run score status pack demo test truth clean-demo

setup:            ## installe Python et les dépendances
	uv sync

check: setup      ## gratuit : clé, plafond, fournisseurs, précisions, prix
	$(UV) loe check

pilot: setup      ## ≈ 0,15 € : stratégie de chaque modèle + 200 points, puis rapport
	-$(UV) loe check
	-$(UV) loe pilot
	$(UV) loe pack
	@printf "\nPousse maintenant results/ sur le repo (voir README, étape 4).\n"

run: setup        ## ≈ 12 € : 16 200 points × 20 modèles ; relancer reprend là où ça s'est arrêté
	$(UV) loe run
	$(UV) loe score
	$(UV) loe pack
	@printf "\nPousse maintenant results/ sur le repo (voir README, étape 6).\n"

score:            ## recalcule classement et cartes à partir des réponses enregistrées
	$(UV) loe score

status:           ## avancement et coût du run en cours (dans un autre terminal)
	$(UV) loe status

pack:             ## compresse les réponses brutes pour le repo
	$(UV) loe pack

demo: setup       ## tout le pipeline contre une fausse API, gratuit, dans results/demo/
	$(UV) loe demo

test: setup       ## tests automatiques (fausse API)
	uv run --quiet pytest -q

truth:            ## reconstruit data/grid.csv (déjà dans le repo, inutile en temps normal)
	uv run --quiet --extra truth python scripts/build_truth.py

clean-demo:
	rm -rf results/demo
