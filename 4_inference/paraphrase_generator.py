import argparse
import logging
import pathlib
from enum import Enum
from typing import List

import pandas
import spacy_universal_sentence_encoder
import transformers
from tqdm.auto import tqdm
from transformers import pipeline, set_seed

from utils import ListDataset

assert transformers.__version__ >= '4.16'


class StrEnum(Enum):
    def __str__(self):
        return self.value


class ParaphraseStrategy(StrEnum):
    TRANSFORMER = "transformer_model"
    BACKTRANSLATION = "backtranslation"


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Paraphrase generator arguments.')
    parser.add_argument('--strategy',
                        type=ParaphraseStrategy,
                        required=True,
                        help='Paraphrase creation strategy.',
                        choices=list(ParaphraseStrategy),
                        default=ParaphraseStrategy.TRANSFORMER)
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='File with sentences to paraphrase. The sentences are seperated by a newline and the file should contain text in UTF-8 encoding')
    parser.add_argument('--out',
                        type=str,
                        help='Output of paraphrased sentences as xlsx file.',
                        default="paraphrases.xlsx")
    args = parser.parse_args()
    strategy = args.strategy
    filename = args.file

    with open(filename, encoding="utf8") as file:
        sentences = [line.strip() for line in file]
        paraphrase_to_excel(sentences, args.out, strategy)


experiment_generate_kwargs = {
    "max_length": 32, 
    "num_beams": 5,
    "early_stopping": True,
}

class ParaphraseGenerator:
    def __init__(self, strategy: ParaphraseStrategy):
        if strategy is ParaphraseStrategy.TRANSFORMER:
            paraphrase_model_path = str(pathlib.Path.home() / "Google Drive" / "mt5" / "mt5-base-paws-x-de") # "GermanT5") 
            self.pipeline_german_paraphrase = pipeline(
                "text2text-generation", 
                model=paraphrase_model_path,
                tokenizer=paraphrase_model_path)
        if strategy is ParaphraseStrategy.BACKTRANSLATION:
            from utils import Translation
            self.translation = Translation()
        
        self.spacy_use = spacy_universal_sentence_encoder.load_model("xx_use_md")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def generate_from_list_of_questions(self, triggers: List[str], num_paraphrases: int = 5) -> List[List[str]]:
        prefix = "paraphrase:"
        # mt5-base-paws-x-de trained without paraphrase prefix 
        formatted_triggers = [f"{trigger} </s>" for trigger in triggers]
        self.logger.debug(f"Generating paraphrases from {len(triggers)} triggers")
        context_dataset = ListDataset(formatted_triggers)
        generated_paraphrases = []
        with tqdm(self.pipeline_german_paraphrase(context_dataset, **experiment_generate_kwargs,
        num_return_sequences=num_paraphrases), # https://github.com/huggingface/huggingface_hub/issues/565 requires version > 4.16
                                                  total=len(context_dataset)) as pbar:
            outputs = list(pbar)
            for out in outputs:
                pbar.set_description("Generating paraphrases")
                generated_paraphrases.append([paraphrase['generated_text'] for paraphrase in out])

        return generated_paraphrases


    def backtranslation(self, examples: List[str]) -> List[List[str]]:
        en_examples = self.translation.translate_de_to_en_multiple_results(examples, 5)
        return self.translation.translate_en_to_de_multiple_results(en_examples, 5)


    def sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        doc1 = self.spacy_use(sentence1)
        doc2 = self.spacy_use(sentence2)
        return float(doc1.similarity(doc2))


def paraphrase_to_excel(text_in: List[str], out_file: str, strategy: ParaphraseStrategy):
    paraphrase_generator = ParaphraseGenerator(strategy)
    paraphrased = None
    if strategy is ParaphraseStrategy.BACKTRANSLATION:
        paraphrased: List[List[str]] = paraphrase_generator.backtranslation(text_in)
    elif strategy is ParaphraseStrategy.TRANSFORMER:
        paraphrased: List[List[str]] = paraphrase_generator.generate_from_list_of_questions(text_in)
    else:
        ValueError("Unknown strategy")
    input_output_list = [(text, paraphrases) for text, paraphrases in list(zip(text_in, paraphrased))]
    input_output = []
    for input, outputs in input_output_list:
        for output in outputs:
            input_output.append((input, output, paraphrase_generator.sentence_similarity(input, output)))
    df = pandas.DataFrame(input_output, columns=["input", "output", "similarity_score"]).reset_index(drop=True)
    df.to_excel(out_file, index=False)


if __name__ == "__main__":
    main()
