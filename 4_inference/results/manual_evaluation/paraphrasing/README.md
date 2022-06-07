# Paraphrase generation
Two approaches to paraphrasing were tested. First, training an mT5 model with the German-language portion of PAWS-X. Second, translation models from huggingface were used for backtranslation.

## Results mT5 finetuning
| No exact copy | Grammatical | Similar meaning | Lexical changes | Syntactic changes |
|---|---|---|---|---|
| 24 | 20 | 6 | 4 | 2 |

## Results backtranslation
| No exact copy | Grammatical | Similar meaning | Lexical changes | Syntactic changes |
|---|---|---|---|---|
| 23 | 20 | 17 | 18 | 3 |

## Google Sheets
In order to better understand the manual evaluation, the google sheets files are provided below.

* [mT5 finetuning](https://docs.google.com/spreadsheets/d/1MCIhyi1miIl9FFzykcRVv7oDGcvqJjMABGiytIkR7ts/edit?usp=sharing)
* [backtranslation](https://docs.google.com/spreadsheets/d/17uqjwcnsnfehDiL24q1t6Aye4wOr2BbgaNmt4lsRIL8/edit?usp=sharing)