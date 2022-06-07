# Answer-aware question generation
This folder contains all files created during the manual evaluation of the answer-aware question generation.

## Results answer-aware question generation
The results of the manual evaluation with 86 questions generated with a trained mT5 model for answer-aware question generation. The model config is contained in `2_training/configs/M-10-mt5-base-hl-qg.json`.

| Webpage (no. questions) / Score | KMI entity text marked (50) | KMI sentence with entity marked (36) |
|---|---|---|
| Ungrammatical or no question | 5 | 4 |
| Grammatical | 26 | 22 |
| Answerable | 19 | 7 |
| Possible chatbot question | 0 | 3 |

## Google Sheets
In order to better understand the manual evaluation, the google sheets files are provided below.

* [KMI answer-aware sentence marked](https://docs.google.com/spreadsheets/d/1sF8iAzh5zYjgsxb5DklS-5xeRSdjXj0rc3AAltmWQkw/edit?usp=sharing)
* [KMI answer-aware text marked](https://docs.google.com/spreadsheets/d/1gUl5kHA8gFUf7cCI82MznaCwaZoVBOPMHtQBnJCaIyg/edit?usp=sharing)