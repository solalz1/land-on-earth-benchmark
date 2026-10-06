# Pilote Land on Earth

24 modèles valides sur 24, 200 points chacun. Coût du pilote : 0.209 $. Projection pour le run complet (16 200 points) : 16.52 $, crédits OpenRouter et clés fournisseurs compris.

Précision estimée : la précision pondérée par la surface que le modèle aurait sur la carte complète, estimée à partir des 200 points (± = intervalle à 90 %). Ce n'est pas encore le score final.

| Modèle | Stratégie | Réflexion | Fournisseur servi | Ta clé | Précision estimée | Couverture | Logprobs | Coût projeté | Statut |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | chat_none | aucune | Novita | — | 89.1% ± 3.7 | 100% | 92% | 0.13 $ | ok |
| DeepSeek V4 Pro | text_none | aucune | SiliconFlow | oui | 82.2% ± 4.7 | 100% | 0% | 0.76 $ | ok |
| Kimi K3 | chat_none | aucune | Alibaba | oui | 89.8% ± 4.7 | 100% | 100% | 4.09 $ | ok |
| Kimi K2.6 | text_none | aucune | SiliconFlow | oui | 84.0% ± 3.9 | 100% | 0% | 0.48 $ | ok |
| GLM-5.3 | text_low | minimale | SiliconFlow | oui | 87.3% ± 4.2 | 100% | 0% | 3.34 $ | ok |
| GLM-5.2 | text_none | aucune | SiliconFlow | oui | 75.6% ± 5.5 | 100% | 0% | 0.77 $ | ok |
| MiniMax M3 | chat_low | minimale | Parasail | — | 91.8% ± 3.4 | 100% | 100% | 2.10 $ | ok |
| Qwen3.5-397B-A17B | chat_none | aucune | Alibaba | oui | 85.7% ± 4.2 | 100% | 100% | 0.27 $ | ok |
| Qwen3.5-122B-A10B | chat_none | aucune | Alibaba | oui | 81.2% ± 5.1 | 100% | 100% | 0.19 $ | ok |
| Qwen3.5-27B | chat_none | aucune | Novita | — | 79.8% ± 5.4 | 100% | 100% | 0.27 $ | ok |
| Qwen3.5-9B | chat_none | aucune | Parasail | — | 29.2% ± 5.3 | 100% | 100% | 0.07 $ | ok |
| Qwen3.6-27B | chat_none | aucune | Alibaba | — | 57.0% ± 6.9 | 100% | 100% | 0.33 $ | ok |
| Qwen3.8-27B | chat_none | aucune | Parasail | — | 55.7% ± 7.3 | 100% | 100% | 0.22 $ | ok |
| Gemma 4 31B | chat_none | aucune | Novita | — | 80.4% ± 5.2 | 100% | 93% | 0.10 $ | ok |
| Gemma 4 26B-A4B | chat_none | aucune | Novita | — | 72.6% ± 5.5 | 100% | 27% | 0.10 $ | ok |
| gpt-oss-120b | chat_low | minimale | Novita | — | 86.2% ± 4.1 | 99% | 47% | 1.23 $ | ok |
| gpt-oss-20b | chat_low | minimale | Novita | — | 78.4% ± 5.3 | 100% | 100% | 0.29 $ | ok |
| Llama 4 Maverick | chat | aucune | Parasail | — | 81.3% ± 5.1 | 100% | 100% | 0.27 $ | ok |
| Nemotron 3 Super | chat_none | aucune | DekaLLM | — | 81.5% ± 4.6 | 100% | 100% | 0.07 $ | ok |
| Ministral 3 3B | text | aucune | Mistral | — | 54.7% ± 7.0 | 100% | 0% | 0.05 $ | ok |
| Ministral 3 8B | text | aucune | Mistral | — | 62.5% ± 6.9 | 100% | 0% | 0.08 $ | ok |
| Ministral 3 14B | text | aucune | Mistral | — | 28.7% ± 5.3 | 100% | 0% | 0.10 $ | ok |
| Mistral Small 4 | text | aucune | Mistral | — | 64.1% ± 7.2 | 100% | 0% | 0.12 $ | ok |
| Mistral Medium 3.5 | text | aucune | Mistral | — | 82.0% ± 4.6 | 100% | 0% | 1.08 $ | ok |

## Stratégies essayées

- **DeepSeek V4.1 Flash** — `chat_none` : ok
- **DeepSeek V4 Pro** — `text_none` : ok
- **Kimi K3** — `chat_none` : ok
- **Kimi K2.6** — `text_none` : ok
- **GLM-5.3** — `text_low` : ok
- **GLM-5.2** — `text_none` : ok
- **MiniMax M3** — `chat_none` : réponse illisible (fin : length) : ''; `chat` : réponse illisible (fin : length) : ''; `raw` : réponse illisible (fin : length) : ''; `text_none` : réponse illisible (fin : length) : ''; `text` : réponse illisible (fin : length) : ''; `chat_low` : ok
- **Qwen3.5-397B-A17B** — `chat_none` : ok
- **Qwen3.5-122B-A10B** — `chat_none` : ok
- **Qwen3.5-27B** — `chat_none` : ok
- **Qwen3.5-9B** — `chat_none` : ok
- **Qwen3.6-27B** — `chat_none` : ok
- **Qwen3.8-27B** — `chat_none` : ok
- **Gemma 4 31B** — `chat_none` : ok
- **Gemma 4 26B-A4B** — `chat_none` : ok
- **gpt-oss-120b** — `chat_none` : 8/8 erreurs — HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled.; `raw` : réponse illisible (fin : length) : ''; `chat_low` : ok
- **gpt-oss-20b** — `chat_none` : 8/8 erreurs — HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled.; `raw` : réponse illisible (fin : length) : ''; `chat_low` : ok
- **Llama 4 Maverick** — `chat` : ok
- **Nemotron 3 Super** — `chat_none` : ok
- **Ministral 3 3B** — `text` : ok
- **Ministral 3 8B** — `text` : ok
- **Ministral 3 14B** — `text` : ok
- **Mistral Small 4** — `text` : ok
- **Mistral Medium 3.5** — `text` : ok
