# Land on Earth — reproduction du tweet

Précision pondérée par la surface contre le masque terre à 1 km (GLOBE). Répondre toujours « Water » donne 71.0 %. Skill = gain sur cette réponse constante. Réflexion « minimale » : le modèle ne peut pas répondre sans réfléchir, il réfléchit au minimum avant de répondre ; les autres répondent sans réflexion.

| # | Modèle | Labo | Réflexion | Précision | Skill | Avec lacs | Rappel terre | Couverture | Logprobs | Stratégie | Fournisseur | Coût |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MiniMax M3 | MiniMax | minimale | 88.8 % | 0.613 | 88.6 % | 92.9 % | 100.0 % | 99.9 % | chat_low | Parasail | 2.05 $ |
| 2 | DeepSeek V4.1 Flash | DeepSeek | aucune | 88.5 % | 0.604 | 88.5 % | 72.7 % | 100.0 % | 77.5 % | chat_none | Novita | 0.10 $ |
| 3 | gpt-oss-120b | OpenAI | minimale | 84.0 % | 0.447 | 83.8 % | 72.2 % | 100.0 % | 26.3 % | chat_low | Novita | 1.02 $ |
| 4 | GLM-5.3 | Z.ai | minimale | 83.6 % | 0.435 | 83.4 % | 93.1 % | 100.0 % | 0.0 % | text_low | SiliconFlow | 3.42 $ |
| 5 | Kimi K3 | Moonshot | aucune | 83.3 % | 0.424 | 83.1 % | 85.3 % | 100.0 % | 100.0 % | chat_none | Alibaba | 4.36 $ |
| 6 | DeepSeek V4 Pro | DeepSeek | aucune | 83.2 % | 0.421 | 83.3 % | 56.4 % | 100.0 % | 0.0 % | text_none | SiliconFlow | 0.76 $ |
| 7 | Kimi K2.6 | Moonshot | aucune | 82.7 % | 0.404 | 82.8 % | 44.2 % | 100.0 % | 0.0 % | text_none | SiliconFlow | 0.48 $ |
| 8 | Mistral Medium 3.5 | Mistral | aucune | 82.0 % | 0.379 | 81.9 % | 50.8 % | 100.0 % | 0.0 % | text | Mistral | 1.27 $ |
| 9 | Nemotron 3 Super | NVIDIA | aucune | 81.3 % | 0.357 | 81.2 % | 65.6 % | 100.0 % | 100.0 % | chat_none | DekaLLM | 0.07 $ |
| 10 | Llama 4 Maverick | Meta | aucune | 80.1 % | 0.316 | 80.1 % | 57.8 % | 100.0 % | 100.0 % | chat | Parasail | 0.26 $ |
| 11 | Qwen3.5-397B-A17B | Alibaba | aucune | 80.0 % | 0.312 | 79.8 % | 78.9 % | 100.0 % | 100.0 % | chat_none | Alibaba | 0.29 $ |
| 12 | Gemma 4 31B | Google | aucune | 79.9 % | 0.307 | 79.6 % | 67.0 % | 100.0 % | 32.0 % | chat_none | Novita | 0.10 $ |
| 13 | gpt-oss-20b | OpenAI | minimale | 79.3 % | 0.288 | 79.2 % | 50.9 % | 100.0 % | 81.0 % | chat_low | Novita | 0.35 $ |
| 14 | Qwen3.5-122B-A10B | Alibaba | aucune | 78.4 % | 0.255 | 78.2 % | 70.9 % | 100.0 % | 100.0 % | chat_none | Alibaba | 0.20 $ |
| 15 | Qwen3.5-27B | Alibaba | aucune | 77.7 % | 0.230 | 77.5 % | 76.1 % | 100.0 % | 100.0 % | chat_none | Novita | 0.27 $ |
| 16 | Gemma 4 26B-A4B | Google | aucune | 74.7 % | 0.129 | 74.8 % | 20.2 % | 100.0 % | 19.1 % | chat_none | Novita | 0.10 $ |
| 17 | GLM-5.2 | Z.ai | aucune | 70.0 % | -0.035 | 69.7 % | 91.2 % | 100.0 % | 0.0 % | text_none | SiliconFlow | 0.77 $ |
| 18 | Ministral 3 8B | Mistral | aucune | 63.6 % | -0.253 | 63.4 % | 71.6 % | 100.0 % | 0.0 % | text | Mistral | 0.08 $ |
| 19 | Mistral Small 4 | Mistral | aucune | 59.6 % | -0.393 | 59.3 % | 82.6 % | 100.0 % | 0.0 % | text | Mistral | 0.12 $ |
| 20 | Qwen3.6-27B | Alibaba | aucune | 55.0 % | -0.549 | 54.8 % | 97.2 % | 100.0 % | 100.0 % | chat_none | Alibaba | 0.33 $ |
| 21 | Qwen3.8-27B | Alibaba | aucune | 51.3 % | -0.678 | 51.0 % | 94.2 % | 100.0 % | 100.0 % | chat_none | Parasail | 0.22 $ |
| 22 | Ministral 3 3B | Mistral | aucune | 50.8 % | -0.693 | 50.6 % | 78.1 % | 100.0 % | 0.0 % | text | Mistral | 0.05 $ |
| 23 | Qwen3.5-9B | Alibaba | aucune | 29.8 % | -1.417 | 29.5 % | 99.8 % | 100.0 % | 100.0 % | chat_none | Parasail | 0.07 $ |
| 24 | Ministral 3 14B | Mistral | aucune | 29.8 % | -1.419 | 29.5 % | 99.7 % | 100.0 % | 0.0 % | text | Mistral | 0.10 $ |
