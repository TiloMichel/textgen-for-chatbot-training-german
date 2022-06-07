# Results
This folder contains the automatic and manual evaluation files mostly in excel format. The sum of the results is provided in readme files. 

The files for automatic evaluation with metrics were generated with files in the folder `3_evaluation`.

The files for manual evaluation were generated with scripts in the folder `4_inference`. 

## The results are organized as follows:
```
📦5_results
 ┣ 📂manual_evaluation
 ┃ ┣ 📂paraphrasing
 ┃ ┃ ┣ 📜backtranslation_de-en-de.xlsx
 ┃ ┃ ┣ 📜mT5_paws-x-de.xlsx
 ┃ ┃ ┗ 📜README.md
 ┃ ┗ 📂question_generation
 ┃ ┃ ┣ 📂answer-agnostic
 ┃ ┃ ┃ ┣ 📜DGE_translation_pipeline_50_questions.xlsx
 ┃ ┃ ┃ ┣ 📜DGE_translation_pipeline_FAQ_50_questions.xlsx
 ┃ ┃ ┃ ┣ 📜KMI_translation_pipeline_48_questions.xlsx
 ┃ ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┃ ┣ 📜RKI_translation_pipeline_50_questions.xlsx
 ┃ ┃ ┃ ┗ 📜RKI_translation_pipeline_FAQ_50_questions.xlsx
 ┃ ┃ ┣ 📂answer-aware
 ┃ ┃ ┃ ┣ 📜KMI_flair_sentence_marked_answer_aware_qg.xlsx
 ┃ ┃ ┃ ┣ 📜KMI_flair_word_marked_answer_aware_qg.xlsx
 ┃ ┃ ┃ ┗ 📜README.md
 ┃ ┃ ┗ 📜README.md
 ┣ 📂metrics
 ┃ ┗ 📂question_generation
 ┃ ┃ ┣ 📂answer-agnostic
 ┃ ┃ ┃ ┣ 📂output
 ┃ ┃ ┃ ┃ ┣ 📂results_english_answer_agnostic_qg
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-2-mt5-base-e2e-qg-squad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂mt5-base-e2e-qg-squad-20ep-adamw
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂mt5-base-e2e-qg-squad-lr5e-4
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂t5-base-e2e-qg
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_gold_reference.txt
 ┃ ┃ ┃ ┃ ┗ 📂results_german_answer_agnostic_qg
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-1-mt5-base-e2e-qg-germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-2-mt5-base-e2e-qg-squad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-3-mt5-base-e2e-qg-translated-squad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-4-mt5-base-e2e-qg-squad-germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-5-mt5-base-e2e-qg-translated-squad-germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-6-mt5-base-e2e-qg-squad+germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-7-mt5-base-e2e-qg-translated-squad+germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂M-9-translation_pipeline
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂mt5-base-e2e-qg-squad+translated-squad+germanquad
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┃ ┃ ┗ 📜xquad_gold_reference.txt
 ┃ ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┃ ┣ 📜results_english_answer_agnostic_qg.xlsx
 ┃ ┃ ┃ ┣ 📜results_german_answer_agnostic_qg.xlsx
 ┃ ┃ ┃ ┗ 📜results_german_translation_pipeline_answer_agnostic_qg.xlsx
 ┃ ┃ ┗ 📂answer-aware
 ┃ ┃ ┃ ┣ 📂output
 ┃ ┃ ┃ ┃ ┣ 📜xquad_gold_reference.txt
 ┃ ┃ ┃ ┃ ┗ 📜xquad_hypothesis.txt
 ┃ ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┃ ┗ 📜results_german_answer_aware_qg.xlsx
 ┗ 📜README.md
 ```