# Results
This folder contains the automatic and manual evaluation files mostly in excel format. The sum of the results is provided in readme files. 

The files for automatic evaluation with metrics were generated with files in the folder `3_evaluation`.

The files for manual evaluation were generated with scripts in the folder `4_inference`. 

## The results are organized as follows:
```
π¦5_results
 β£ πmanual_evaluation
 β β£ πparaphrasing
 β β β£ πbacktranslation_de-en-de.xlsx
 β β β£ πmT5_paws-x-de.xlsx
 β β β πREADME.md
 β β πquestion_generation
 β β β£ πanswer-agnostic
 β β β β£ πDGE_translation_pipeline_50_questions.xlsx
 β β β β£ πDGE_translation_pipeline_FAQ_50_questions.xlsx
 β β β β£ πKMI_translation_pipeline_48_questions.xlsx
 β β β β£ πREADME.md
 β β β β£ πRKI_translation_pipeline_50_questions.xlsx
 β β β β πRKI_translation_pipeline_FAQ_50_questions.xlsx
 β β β£ πanswer-aware
 β β β β£ πKMI_flair_sentence_marked_answer_aware_qg.xlsx
 β β β β£ πKMI_flair_word_marked_answer_aware_qg.xlsx
 β β β β πREADME.md
 β β β πREADME.md
 β£ πmetrics
 β β πquestion_generation
 β β β£ πanswer-agnostic
 β β β β£ πoutput
 β β β β β£ πresults_english_answer_agnostic_qg
 β β β β β β£ πM-2-mt5-base-e2e-qg-squad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πmt5-base-e2e-qg-squad-20ep-adamw
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πmt5-base-e2e-qg-squad-lr5e-4
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πt5-base-e2e-qg
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β πxquad_gold_reference.txt
 β β β β β πresults_german_answer_agnostic_qg
 β β β β β β£ πM-1-mt5-base-e2e-qg-germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-2-mt5-base-e2e-qg-squad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-3-mt5-base-e2e-qg-translated-squad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-4-mt5-base-e2e-qg-squad-germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-5-mt5-base-e2e-qg-translated-squad-germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-6-mt5-base-e2e-qg-squad+germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-7-mt5-base-e2e-qg-translated-squad+germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πM-9-translation_pipeline
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β£ πmt5-base-e2e-qg-squad+translated-squad+germanquad
 β β β β β β β πxquad_hypothesis.txt
 β β β β β β πxquad_gold_reference.txt
 β β β β£ πREADME.md
 β β β β£ πresults_english_answer_agnostic_qg.xlsx
 β β β β£ πresults_german_answer_agnostic_qg.xlsx
 β β β β πresults_german_translation_pipeline_answer_agnostic_qg.xlsx
 β β β πanswer-aware
 β β β β£ πoutput
 β β β β β£ πxquad_gold_reference.txt
 β β β β β πxquad_hypothesis.txt
 β β β β£ πREADME.md
 β β β β πresults_german_answer_aware_qg.xlsx
 β πREADME.md
 ```