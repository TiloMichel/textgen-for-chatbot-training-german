# Model inference
This folder contains all scripts to run models and create excel files for manual evaluation.

## Question generation
To be able to use question generation, previous text extraction of a web page is required (see folder `0_webcrawler`)

Check all possible arguments with:

`python question_generator.py --help`

The following commands are available.
Generate questions from website (answer-agnostic, translation pipeline)

`python question_generator.py --qg-strategy answer_agnostic --textblock-strategy all --index_name kmi_crawler_2364dfd1-a060-44d9-85a8-eb7e5ecc3fe1 --translation-pipeline True`

### Other inference commands
Generate questions from website (answer-agnostic):

`python question_generator.py --qg-strategy answer_agnostic --textblock-strategy all --index_name kmi_crawler_2364dfd1-a060-44d9-85a8-eb7e5ecc3fe1`

Generate questions from faq website (answer-agnostic):

`python question_generator.py --qg-strategy answer_agnostic --textblock-strategy faq --index_name kmi_crawler_2364dfd1-a060-44d9-85a8-eb7e5ecc3fe1`

Generate questions from website (answer-aware):

`python question_generator.py --qg-strategy answer_aware --textblock-strategy all --index_name kmi_crawler_2364dfd1-a060-44d9-85a8-eb7e5ecc3fe1 --ner-library flair --ner-highlighting-strategy word`

## Paraphrasing
For paraphrasing there is a file in this directory named `training_examples.txt` which is used for inference. If you want to paraphrase other sentences create a new text file and provide its location via CLI.

Check all possible arguments with:

`python paraphrase_generator.py --help`

The following commands are available.

With transformer model:

`python paraphrase_generator.py --strategy transformer_model --file training_examples.txt --out paraphrases1.xlsx`

With backtranslation:

`python paraphrase_generator.py --strategy backtranslation --file training_examples.txt --out paraphrases1.xlsx`