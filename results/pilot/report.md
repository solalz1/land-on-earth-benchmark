# Pilote Land on Earth

5 modèles valides sur 20, 200 points chacun. Coût du pilote : 0.437 $. Projection pour le run complet (16 200 points) : 35.37 $ de crédits.

| Modèle | Stratégie | Fournisseur servi | Précision | Couverture | Masse Land+Water | Coût projeté | Statut |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | chat_none | Novita | 76.0% | 100% | 1.00 | 0.13 $ | à vérifier : logprobs sur 92% des réponses |
| DeepSeek V4 Pro | chat_low | Novita | 97.3% | 74% | 1.00 | 26.95 $ | à vérifier : couverture 74%, coût projeté 26.95 $ contre 1.63 $ estimé |
| Kimi K3 | chat_none | Parasail | 78.1% | 94% | 0.98 | 3.91 $ | à vérifier : couverture 94% |
| Kimi K2.6 | chat_none | Parasail | 68.5% | 92% | 1.00 | 0.48 $ | à vérifier : couverture 92% |
| GLM-5.3 | — | — | — | — | — | — | échec |
| GLM-5.2 | — | — | — | — | — | — | échec |
| MiniMax M3 | chat_low | Parasail | 87.5% | 100% | 1.00 | 2.10 $ | à vérifier : coût projeté 2.10 $ contre 0.32 $ estimé |
| Qwen3.5-397B-A17B | chat_none | Parasail | 70.8% | 92% | 1.00 | 0.40 $ | à vérifier : couverture 92% |
| Qwen3.5-122B-A10B | — | — | — | — | — | — | échec |
| Qwen3.5-27B | — | — | — | — | — | — | échec |
| Qwen3.5-9B | chat_none | Parasail | 49.5% | 100% | 1.00 | 0.07 $ | ok |
| Qwen3.6-27B | — | — | — | — | — | — | échec |
| Qwen3.8-27B | chat_none | Parasail | 63.5% | 100% | 1.00 | 0.22 $ | ok |
| Gemma 4 31B | chat_none | Novita | 70.5% | 100% | 1.00 | 0.10 $ | à vérifier : logprobs sur 93% des réponses |
| Gemma 4 26B-A4B | raw | Novita | 52.5% | 100% | 1.00 | 0.13 $ | à vérifier : logprobs sur 26% des réponses |
| gpt-oss-120b | — | — | — | — | — | — | échec |
| gpt-oss-20b | chat_low | Novita | 69.0% | 100% | 1.00 | 0.29 $ | ok |
| Llama 4 Maverick | chat | Parasail | 64.5% | 100% | 0.82 | 0.27 $ | ok |
| Nemotron 3 Super | chat_none | DekaLLM | 66.0% | 100% | 0.98 | 0.07 $ | ok |
| Mistral Large 3 | text | Mistral | 58.2% | 79% | — | 0.25 $ | à vérifier : couverture 79% |

## Stratégies essayées

- **DeepSeek V4.1 Flash** — `chat_none` : ok
- **DeepSeek V4 Pro** — `chat_none` : pas de logprobs exploitables dans la réponse; `chat` : réponse illisible (fin : length) : ''; `raw` : template indisponible (dépôt Hugging Face introuvable ou protégé); `chat_low` : ok
- **Kimi K3** — `chat_none` : ok
- **Kimi K2.6** — `chat_none` : ok
- **GLM-5.3** — `chat_none` : 8/8 erreurs — HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled.; `chat` : réponse illisible (fin : length) : ''; `raw` : réponse illisible (fin : length) : ''; `chat_low` : pas de logprobs exploitables dans la réponse
- **GLM-5.2** — `chat_none` : pas de logprobs exploitables dans la réponse; `chat` : le modèle raisonne malgré la consigne; `raw` : réponse illisible (fin : length) : 'The coordinates 51.0° N'; `chat_low` : pas de logprobs exploitables dans la réponse
- **MiniMax M3** — `chat_none` : réponse illisible (fin : length) : ''; `chat` : réponse illisible (fin : length) : ''; `raw` : réponse illisible (fin : length) : ''; `chat_low` : ok
- **Qwen3.5-397B-A17B** — `chat_none` : ok
- **Qwen3.5-122B-A10B** — `chat_none` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: d323a79b54ec110a585e5a365085af4e","type":"invalid_request_error"}
; `chat` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: ac78c4e9c41713c9b5e3ce78dea3502f","type":"invalid_request_error"}
; `raw` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: 7dbaf6e8cf0128fed9b10e4b60549e92","type":"invalid_request_error"}
; `chat_low` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: cb6c7b412821fbcc958d0c18e8c6dc9c","type":"invalid_request_error"}

- **Qwen3.5-27B** — `chat_none` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: 72cff542a20bad270c0d7e54dbfbc40e","type":"invalid_request_error"}
; `chat` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: 2c7e3dbab9cfd422c83c8c89622d8b48","type":"invalid_request_error"}
; `raw` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: 0dc28bd4791281c5c3c1fcfcb7df4fc7","type":"invalid_request_error"}
; `chat_low` : 8/8 erreurs — HTTP 400: Provider returned error | {"message":"invalid request error trace_id: 2ce81cb32b0ffdf678d6cc7d4314cf94","type":"invalid_request_error"}

- **Qwen3.5-9B** — `chat_none` : ok
- **Qwen3.6-27B** — `chat_none` : 8/8 erreurs — HTTP 400: Provider returned error | data: {"error":{"code":"invalid_parameter_error","param":null,"message":"Range of top_logprobs should be [0, 5]","type":"invalid_request_error"},"id":"chatcmpl-54036480-037a-9e3e-b1fe-50453ba831a1"}

; `chat` : 8/8 erreurs — HTTP 400: Provider returned error | data: {"error":{"code":"invalid_parameter_error","param":null,"message":"Range of top_logprobs should be [0, 5]","type":"invalid_request_error"},"id":"chatcmpl-1ff1a815-00d8-9337-b985-0d1e68224b82"}

; `raw` : 8/8 erreurs — HTTP 400: Provider returned error | data: {"error":{"code":"invalid_parameter_error","param":null,"message":"Range of top_logprobs should be [0, 5]","type":"invalid_request_error"},"id":"chatcmpl-9ace477b-3477-957f-a158-fed7ed738de2"}

; `chat_low` : 8/8 erreurs — HTTP 400: Provider returned error | data: {"error":{"code":"invalid_parameter_error","param":null,"message":"Range of top_logprobs should be [0, 5]","type":"invalid_request_error"},"id":"chatcmpl-b9ed3e24-d002-954c-a989-2f75eb01b6cf"}


- **Qwen3.8-27B** — `chat_none` : ok
- **Gemma 4 31B** — `chat_none` : ok
- **Gemma 4 26B-A4B** — `chat_none` : pas de logprobs exploitables dans la réponse; `chat` : pas de logprobs exploitables dans la réponse; `raw` : ok
- **gpt-oss-120b** — `chat_none` : 8/8 erreurs — HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled.; `raw` : réponse illisible (fin : length) : ''; `chat_low` : pas de logprobs exploitables dans la réponse
- **gpt-oss-20b** — `chat_none` : 8/8 erreurs — HTTP 400: Reasoning is mandatory for this endpoint and cannot be disabled.; `raw` : réponse illisible (fin : length) : ''; `chat_low` : ok
- **Llama 4 Maverick** — `chat` : ok
- **Nemotron 3 Super** — `chat_none` : ok
- **Mistral Large 3** — `text` : ok
