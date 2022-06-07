# Answer-agnostic question generation metrics
This folder contains all results from metrics for answer-agnostic question generation. The results were calculated with the Jupyter Notebooks `3_evaluation/multiple_model_evaluation.ipynb` and `3_evaluation/translation_pipeline_evaluation.ipynb`.

## Results
The two tables below show the results of automatic metrics.
The following text style was used for the best measurements:
* **number** denotes the best model. 
* *number* denotes the best trained Model.

### Test of English trained models

| Model | Hyperparameter changes | BLEU Score | ROUGE-L (f-measure) | METEOR-score | BERT Score (f-measure) |
|---|---|---|---|---|---|
| mT5 (M-2) | LR 7e-4, 10 Epochs | 7.17 | 0.2865 | 0.1034 | 0.2957 |
| mT5 | - | 7.22 | 0.2884 | 0.1063 | 0.3063 |
| mT5 | LR 5e-4, 10 Epochs | 7.44 | 0.2880 | 0.1061 | 0.2987 |
| T5 \cite{2020Patil} | - | **13.59** | **0.3233** | **0.1413** | **0.3581** |

### Test of German models

| Model | BLEU Score | ROUGE-L (f-measure) | METEOR-score | BERT Score |
|---|---|---|---|---|
| M-1 | 1.73 | 0.1711 | 0.0835 | 0.3319 |
| M-2 | 0.83 | 0.1016 | 0.0348 | 0.2244 |
| M-3 | *4.18* | 0.1986 | *0.1132* | 0.3347 |
| M-4 | 1.98 | 0.1807 | 0.0912 | 0.3388 |
| M-5 | 2.29 | 0.1899 | 0.0942 | 0.3408 |
| M-6 | 2.10 | 0.1828 | 0.0909 | *0.3449* |
| M-7 | 3.81 | *0.2014* | 0.1119 | 0.3379 |
| M-9 | **7.5** | **0.2185** | **0.1447** | **0.3481** |