from enum import Enum
from utils import Translation, ListDataset
from question_answer import (WebpageText, QuestionsAnswerItem, Textblock, AnswerItem, Textblock_with_entity)
from typing import Dict, Iterator, List, Any, Set, Tuple
from elasticsearch import Elasticsearch
from spacy.tokens import Doc
import language_tool_python
import logging
from tqdm.auto import tqdm
import re
from flair.tokenization import SegtokSentenceSplitter
import random
from transformers import set_seed, pipeline
import spacy
import pathlib
import torch
from flair.models import SequenceTagger
import flair
import argparse
import pandas


class StrEnum(Enum):
    def __str__(self):
        return self.value


class QuestionGenerationStrategy(StrEnum):
    answer_agnostic = "answer_agnostic"
    answer_aware = "answer_aware"


class TextblockStrategy(StrEnum):
    all = "all"
    faq = "faq"


class NerLibrary(StrEnum):
    spaCy = "spaCy"
    flair = "flair"


class NerHighlightingStrategy(StrEnum):
    word = "word"
    sentence = "sentence"


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Paraphrase generator arguments.')
    parser.add_argument('--qg-strategy',
                        type=QuestionGenerationStrategy,
                        required=True,
                        help='Question generation strategy.',
                        choices=list(QuestionGenerationStrategy),
                        default=QuestionGenerationStrategy.answer_agnostic)
    parser.add_argument('--textblock-strategy',
                        type=TextblockStrategy,
                        required=True,
                        help='Used textblock strategy.',
                        choices=list(TextblockStrategy),
                        default=TextblockStrategy.all)
    parser.add_argument('--ner-library',
                        type=NerLibrary,
                        help='NER highlighting strategy.',
                        choices=list(NerLibrary),
                        default=NerLibrary.flair)
    parser.add_argument('--ner-highlighting-strategy',
                        type=NerHighlightingStrategy,
                        help='NER highlighting strategy.',
                        choices=list(NerHighlightingStrategy),
                        default=NerHighlightingStrategy.word)
    parser.add_argument('--index-name',
                        type=str,
                        required=True,
                        help='Elasticsearch index name for websites.')
    parser.add_argument('--random-samples',
                        type=bool,
                        help='Use random webpages.',
                        default=False)
    parser.add_argument('--random-samples-count',
                        type=int,
                        help='Elasticsearch index name for websites.',
                        default=3)
    parser.add_argument('--translation-pipeline',
                        type=bool,
                        help='Use a translation answer-agnostic question generation pipeline.',
                        default=False)

    args = parser.parse_args()
    qg_strategy = args.qg_strategy
    ner_lib = args.ner_library
    ner_hl_strat = args.ner_highlighting_strategy
    index_name = args.index_name
    textblock_strategy = args.textblock_strategy
    random_samples = args.random_samples
    random_samples_websites_count = args.random_samples_count
    use_translation_pipeline = args.translation_pipeline

    question_generator = QuestionGenerator()
    answers = questions = None
    if qg_strategy == QuestionGenerationStrategy.answer_aware:
        if not ner_lib:
            raise ValueError("Named entity recognition Library muss gesetzt sein")
        if not ner_hl_strat:
            raise ValueError("Named entity recognition highlighting strategy muss gesetzt sein")
        answers, questions = question_generator.generate_from_highlighted_ner_index(
            index_name,
            ner_lib,
            ner_hl_strat,
            textblock_strategy,
            random_samples,
            random_samples_websites_count)
    elif qg_strategy == QuestionGenerationStrategy.answer_agnostic:
        answers, questions = question_generator.generate_from_index(
            index_name,
            textblock_strategy,
            use_translation_pipeline,
            random_samples,
            random_samples_websites_count)
    else:
        raise ValueError("Question generation strategy not set")
    answers_df = pandas.DataFrame([vars(a) for a in answers])
    questions_df = pandas.DataFrame([vars(q) for q in questions])
    print(answers_df.head())
    print(questions_df.head())
    filename = "output.xlsx"
    if qg_strategy == QuestionGenerationStrategy.answer_aware:
        if random_samples:
            filename = f"{index_name}_{textblock_strategy}_{ner_lib}_{ner_hl_strat}_{qg_strategy}_{random_samples_websites_count}_samples_qa.xlsx"
        else:
            filename = f"{index_name}_{textblock_strategy}_{ner_lib}_{ner_hl_strat}_{qg_strategy}_qa.xlsx"
    elif qg_strategy == QuestionGenerationStrategy.answer_agnostic:
        if random_samples:
            filename = f"{index_name}_{textblock_strategy}_{qg_strategy}_{random_samples_websites_count}_samples_qa.xlsx"
        else:
            filename = f"{index_name}_{textblock_strategy}_{qg_strategy}_qa.xlsx"
    if use_translation_pipeline:
        filename = f"{index_name}_{textblock_strategy}_{qg_strategy}_translation_pipeline_qa.xlsx"
    with pandas.ExcelWriter(filename) as writer:
        questions_df.to_excel(writer, sheet_name='question_answer_pairs')
        answers_df.to_excel(writer, sheet_name='answers')


class ElasticSearchClient:
    def __init__(self) -> None:
        # requires default installation of elasticsearch no password
        # otherwise changes have to be made here
        self.es = Elasticsearch(timeout=720)

    def match_all_query(self) -> Dict[str, Any]:
        return {
            "size": 100,
            "query": {"match_all": {}},
            "sort": [{"added_position": "asc"}]
        }


    def match_faq_pages(self, search_term: str = "faq", default_field: str = "url") -> Dict[str, Any]:
        return {
            "size": 100,
            "query": {
                "query_string": {
                    "query": search_term,
                    "default_field": default_field
                }
            },
            "sort": [{"added_position": "asc"}]
        }

    def query_search_after(self, index_name: str, body: Dict) -> Iterator[WebpageText]:
        while True:
            res = self.es.search(index=index_name, body=body)
            results_len = len(res["hits"]["hits"])
            if results_len <= 0:
                break
            for hit in res["hits"]["hits"]:
                dict = hit["_source"]
                yield WebpageText(
                    title=dict["title"],
                    extracted_text=dict["text"],
                    depth=dict["depth"],
                    url=dict["url"],
                    added_position=dict["added_position"]
                )
            body["search_after"] = [
                res["hits"]["hits"][-1]["_source"]["added_position"]
            ]


    def get_website_text_from_index(self, index_name: str) -> Iterator[WebpageText]:
        return self.query_search_after(index_name, self.match_all_query())


    def get_faq_website_text_from_index(self, index_name: str, search_term: str = "faq", default_field: str = "url") -> Iterator[WebpageText]:
        return self.query_search_after(index_name, self.match_faq_pages(search_term=search_term, default_field=default_field))


class QuestionGenerator:
    def __init__(self):
        model_path = str(pathlib.Path.home() / "Google Drive" / "mt5" / "M-6-mt5-base-e2e-qg-squad+germanquad")
        self.pipeline_german_e2e_question_generation = pipeline(
            "text2text-generation",
            model=model_path,
            tokenizer=model_path)

        hl_qg_model_path = str(pathlib.Path.home() / "Google Drive" / "mt5" / "M-9-mt5-base-hl-qg-germanquad")
        self.pipeline_german_hl_question_generation =  pipeline(
            "text2text-generation", 
            model=hl_qg_model_path,
            tokenizer=hl_qg_model_path)
        self.spacy = spacy.load("de_core_news_lg")

        flair.device = torch.device("cpu") # overwrite device (CPU inference)
        self.flair_ner_tagger = SequenceTagger.load("flair/ner-german-large")

        qg_e2e_model = "valhalla/t5-base-e2e-qg"
        self.en_qg_pipeline = pipeline("text2text-generation", model=qg_e2e_model, tokenizer=qg_e2e_model)
        self.translation = Translation()
        self.language_tool = language_tool_python.LanguageTool('de-DE')
        self.es_client = ElasticSearchClient()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def translation_qg_pipeline(self, textblocks: List[Textblock]) ->  List[Tuple[Textblock, List[str]]]:
        """
        Using a text, try to formulate a question that fits the context of the text.

        :param text: input string to generate question from
        :return: question which relates to the input context
        """
        texts = [textblock.text for textblock in textblocks]
        texts_en = self.translation.translate_de_to_en(texts)
        texts_en_formatted = [f"generate question: {text} </s>" for text in texts_en]
        generated_questions_en = self.en_qg_pipeline(texts_en_formatted,
                                                       batch_size=1,
                                                       max_length=128,
                                                       num_beams=4,
                                                       length_penalty=1.5,
                                                       no_repeat_ngram_size=3,
                                                       early_stopping=True)
        generated_questions_en = [text['generated_text'] for text in generated_questions_en] # extract questions
        generated_questions_en = [text.split("<sep>") for text in generated_questions_en] # split questions into lists
        context_questions = []
        for question_list in generated_questions_en:
            context_questions.append([question.strip() for question in question_list if question])
        
        translated_questions = []
        for questions in context_questions:
            translated_questions.append(self.translation.translate_en_to_de(questions))
        
        assert len(textblocks) == len(translated_questions)
        textblock_qlist = [(textblock, qlist) for textblock, qlist in zip(textblocks, translated_questions)]
        return textblock_qlist


    def __ends_with_question_mark(self, text_doc: Doc) -> bool:
        """Checks if a text ends with an question mark

        The Grammar tagger from spacy is used see:
        Stuttgart-Tübingen-Tagset STTS-Tags
        https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger

        Args:
            text (Doc): Text to determine from if it ends with a question mark

        Returns:
            bool: True if question mark is the last token and ends the sentence else False
        """
        SENTENCE_PUNCTUATION = "$."
        return (text_doc[-1].tag_ == SENTENCE_PUNCTUATION and 
                text_doc[-1].text == "?")


    def __contains_at_least_one_interrogative_pronoun(self, text_doc: Doc) -> bool:
        """Method to determine if a text contains one or more interrogative pronouns

        The Grammar tagger from spacy is used see:
        Stuttgart-Tübingen-Tagset STTS-Tags
        https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger
        PWS substituierendes Interrogativpronomen 'wer, was'
        PWAT attributierendes Interrogativpronomen 'welche, wessen'
        PWAV adverbiales Interrogativpronomen oder Relativpronomen 'warum, wo, wann, worüber, wobei'

        Args:
            text (Doc): Text to determine from if it contains a interrogative pronoun

        Returns:
            bool: True if contains at least one interrogative pronoun else False
        """
        INTERROGATIVPRONOMEN_STTS_TAGS = ["PWS", "PWAT", "PWAV"]
        interrogative_pronouns = []
        for token in text_doc: # Nicht immer am Anfang des Satzes
            if token.tag_ in INTERROGATIVPRONOMEN_STTS_TAGS: 
                interrogative_pronouns.append(token.text)

        return len(interrogative_pronouns) >= 1


    def __is_question(self, text_doc: Doc) -> bool:
        """Classify a text as question if it ends with an question mark and it contains a interrogative pronoun

        Args:
            text (Doc): Text to classify

        Returns:
            bool: True if question else False
        """
        return self.__ends_with_question_mark(text_doc) and self.__contains_at_least_one_interrogative_pronoun(text_doc)

    
    def __contains_at_least_one_sentence(self, text_doc: Doc) -> bool:
        at_least_one_correct_sentence = False
        for sentence in text_doc.sents:
            if self.__has_subject_predicate_or_verb_object(sentence):
                at_least_one_correct_sentence = True
                break
        
        return at_least_one_correct_sentence


    def __has_subject_predicate_or_verb_object(self, sentence: Doc) -> bool:
        """Returns if a text is a sentence. This means whether it contains subject predicate or verb and object.

        Args:
            text_doc (Doc): Text to classify

        Returns:
            bool: True if the text is a sentence else False
        """
        # https://v2.spacy.io/api/annotation#dependency-parsing-german
        # TIGER Treebank dependency parse format
        subject_dep_labels = ["sb", "sbp", "sp"] # subject (sb), passivized subject (PP) (sbp), subject or predicate (sp)
        object_dep_labels = ["oa", "oa2", "oc", "og", "op"] # accusative object (oa), second accusative object (oa2). clausal object (oc), genetive object (og), prepositional object (op)
        predicate_dep_labels = ["sp", "pd"] # subject or predicate (sp), predicate (pd)
        subject = predicate_or_verb = obj = None
        for token in sentence:
            dep = token.dep_
            if dep in subject_dep_labels:
                subject = (token.text, dep)
            if dep in object_dep_labels:
                obj = (token.text, dep)
            # Prädikate werden vom dependency parser kaum erkannt, deshalb werden auch Verben verwendet 
            if (dep in predicate_dep_labels) or token.pos_ == "VERB":
                predicate_or_verb = (token.text, dep)

        return subject and predicate_or_verb and obj


    def __get_distinct_textblocks_from_webpage_texts(self, webpages: List[WebpageText]) -> List[Textblock]:
        distinct_extracted_text = []
        distinct_textblocks = []
        # get distinct textblocks from all webpages
        for webpage in webpages:
            if webpage.extracted_text not in distinct_extracted_text:
                distinct_extracted_text.append(webpage.extracted_text)
            else:
                self.logger.debug(f"{webpage.url} skipped: possible duplicate")
                continue 
            paragraph_position = 0
            textblocks = [textblock.strip().replace("\n", " ") for textblock in webpage.extracted_text.split("\n\n") if textblock] # blocks are split by double newline in intentfindercrawler spider
            for textblock in textblocks:
                if textblock not in distinct_textblocks:
                    if not textblock:
                        self.logger.debug(f"Empty textblock")
                        continue
                    doc = self.spacy(textblock)
                    if self.__is_question(doc):
                        self.logger.debug(f"The textblock seems to be a question and is not added as an answer: {textblock}")
                    elif not self.__contains_at_least_one_sentence(doc):
                        self.logger.debug(f"The textblock seems to contain no correct sentence (at least one sentence with subject, predicate/verb and object): {textblock}")
                    else:
                        distinct_textblocks.append(Textblock(
                            title=webpage.title, 
                            url=webpage.url, 
                            depth= webpage.depth, 
                            text=textblock,
                            page_added_position=webpage.added_position, 
                            page_text_position=paragraph_position))
                        paragraph_position += 1
        
        return distinct_textblocks


    def __generate_questions_end_to_end_from_textblocks(self, textblocks: List[Textblock]) -> List[Tuple[Textblock, List[str]]]:
        formatted_paragraphs = [f"generate question: {textblock.text} </s>" for textblock in textblocks]
        self.logger.debug(f"Generating questions from {len(formatted_paragraphs)} textblocks")
        context_dataset = ListDataset(formatted_paragraphs)
        generated_questions = []
        with tqdm(self.pipeline_german_e2e_question_generation(context_dataset,
                                                               batch_size=1,
                                                               max_length=128,
                                                               num_beams=4,
                                                               length_penalty=1.5,
                                                               no_repeat_ngram_size=3,
                                                               early_stopping=True,
                                                               ),
                                                               total=len(context_dataset)) as pbar:
            for out in pbar:
                pbar.set_description("Generating questions")
                generated_questions.append(out)

        generated_questions = [gen_question[0]['generated_text'] for gen_question in generated_questions]
        generated_questions = [question_list.split("<sep>") for question_list in generated_questions]

        assert len(textblocks) == len(generated_questions)
        question_list_per_answer: List[Tuple[Textblock, List[str]]] = []
        for textblock, question_list in list(zip(textblocks, generated_questions)): # list of lists
            q_list = []
            for question in question_list:
                stripped_question = question.strip()
                if stripped_question not in q_list: # filter duplicate questions
                    if not stripped_question:
                        self.logger.debug(f"Empty textblock")
                        continue
                    if self.__is_question(self.spacy(stripped_question)):
                        q_list.append(stripped_question)
                    else:
                        self.logger.debug(f"not a question: {stripped_question}")
                else:
                    self.logger.debug(f"skipped duplicate question: {stripped_question}")
            question_list_per_answer.append((textblock, q_list))

        return question_list_per_answer


    def highlight_named_entity_in_textblock(
        self,
        textblock: str,
        ner_text: str,
        highlight_token: str = "<hl>") -> List[Tuple[str, str]]:
        assert type(textblock) == str and type(ner_text) == str
        regex_ner_text = re.escape(ner_text)
        regex_ner_text = rf"\b{regex_ner_text}\b"
        highlighted_contexts: List[Tuple[str, str]] = []
        # replace the entity text for each occurence in the textblock
        for match in re.finditer(regex_ner_text, textblock):
            end, newstart = match.span()
            highlighted_textblock = textblock[0:end] # start bis entity text
            text_to_highlight = match.group(0).strip()
            highlighted_textblock += f"{highlight_token}{text_to_highlight}{highlight_token}"
            highlighted_textblock += textblock[newstart:] # entity text ende bis ende textblock
            highlighted_contexts.append((highlighted_textblock, text_to_highlight))

        return highlighted_contexts


    def highlight_sentence_with_named_entity_in_textblock(
        self,
        textblock: str,
        ner_text: str,
        highlight_token: str = "<hl>") -> List[Tuple[str, str]]:
        assert type(textblock) == str and type(ner_text) == str
        doc = self.spacy(textblock)
        textblock_sentence_highlighted_per_occurence = []
        entity_occurence_count: int = textblock.count(ner_text)

        for occurence_count in range(1, entity_occurence_count + 1):
            highlighted_textblock = ""
            found_ner_count = 0
            for sent in doc.sents:
                if ner_text in sent.text:
                    found_ner_count += 1
                    if occurence_count == found_ner_count:
                        highlighted_textblock += f"{highlight_token}{sent.text}{highlight_token}"
                    else:
                        highlighted_textblock += sent.text
                else:
                    highlighted_textblock += sent.text

            found_sentence = [hl_textblock 
            for hl_textblock, _ in textblock_sentence_highlighted_per_occurence 
            if hl_textblock == highlighted_textblock] # find duplicates in list of tuples
            if not found_sentence: # skip duplicates (sentences with multiple entities)
                textblock_sentence_highlighted_per_occurence.append((highlighted_textblock, ner_text))
            else:
                self.logger.debug(f"Skipping duplicate marked sentence {found_sentence}")

        return textblock_sentence_highlighted_per_occurence


    def language_tool_check_and_correct(self, text: str) -> Tuple[str, List[Any]]:
        matches = self.language_tool.check(text)
        # Probleme mit Eigenwörtern, Abkürzungen und Fremdsprache (Englisch) bei Autokorrektur
        matches_no_spelling_errors = [match for match in matches if match.ruleId not in ["GERMAN_SPELLER_RULE", "ZAHL_IM_WORT"]]

        if matches_no_spelling_errors:
            corrected_text = language_tool_python.utils.correct(text, matches_no_spelling_errors)
            return corrected_text, matches

        return text, matches


    # main end-to-end question generation
    def generate_from_index(self, 
                            elastic_search_index_name: str, 
                            textblock_strategy: TextblockStrategy,
                            use_translation_pipeline: bool,
                            random_samples: bool,
                            random_samples_websites_count: int) -> Tuple[List[AnswerItem], List[QuestionsAnswerItem]]:
        webpage_items = None
        if textblock_strategy == textblock_strategy.all:
            if random_samples:
                webpage_items = random.sample(list(self.es_client.get_website_text_from_index(elastic_search_index_name)), random_samples_websites_count)
            else:
                webpage_items = list(self.es_client.get_website_text_from_index(elastic_search_index_name))
        elif textblock_strategy == textblock_strategy.faq:
            if random_samples:
                webpage_items = random.sample(list(self.es_client.get_faq_website_text_from_index(elastic_search_index_name)), random_samples_websites_count)
            else:
                webpage_items = list(self.es_client.get_faq_website_text_from_index(elastic_search_index_name))

        logging.debug(f"Webpage count: {len(webpage_items)}")

        distinct_textblocks = self.__get_distinct_textblocks_from_webpage_texts(webpage_items)
        answer_items = []
        for textblock in distinct_textblocks:
            answer_items.append(AnswerItem.copy_from(textblock))
        
        textblock_questions = None
        if use_translation_pipeline:
            textblock_questions = self.translation_qg_pipeline(distinct_textblocks)
        else:
            textblock_questions = self.__generate_questions_end_to_end_from_textblocks(distinct_textblocks)

        question_answer_items = []
        for paragraph, questions in tqdm(textblock_questions, total=len(textblock_questions)):
            for question in questions:
                question, matches = self.language_tool_check_and_correct(question)
                question_answer_items.append(QuestionsAnswerItem.copy_from(paragraph, question, matches))

        return answer_items, question_answer_items

    
    # ner answer-aware question generation
    def generate_from_highlighted_ner_index(self, 
                                            elastic_search_index_name: str, 
                                            ner_lib: NerLibrary, 
                                            highlighting_strategy: NerHighlightingStrategy,
                                            textblock_strategy: TextblockStrategy,
                                            random_samples: bool,
                                            random_samples_websites_count: int) -> Tuple[List[AnswerItem], List[QuestionsAnswerItem]]:
        webpage_items = None
        if textblock_strategy == textblock_strategy.all:
            if random_samples:
                webpage_items = random.sample(list(self.es_client.get_website_text_from_index(elastic_search_index_name)), random_samples_websites_count)
            else:
                webpage_items = list(self.es_client.get_website_text_from_index(elastic_search_index_name))
        elif textblock_strategy == textblock_strategy.faq:
            if random_samples:
                webpage_items = random.sample(list(self.es_client.get_faq_website_text_from_index(elastic_search_index_name)), random_samples_websites_count)
            else:
                webpage_items = list(self.es_client.get_faq_website_text_from_index(elastic_search_index_name))

        logging.debug(f"Webpage count: {len(webpage_items)}")
        answer_items = []
        textblocks = self.__get_distinct_textblocks_from_webpage_texts(webpage_items)
        for textblock in textblocks:
            answer_items.append(AnswerItem.copy_from(textblock))

        textblock_questions = self.__generate_questions_from_textblocks_ner_highlighted_qg(textblocks, ner_lib, highlighting_strategy)
        
        question_answer_items = []
        for paragraph, question in tqdm(textblock_questions, total=len(textblock_questions)):
            question, matches = self.language_tool_check_and_correct(question)
            question_answer_items.append(QuestionsAnswerItem.copy_from(paragraph, question, matches))

        return answer_items, question_answer_items


    def __get_textblock_with_entities_flair(self, textblocks: List[Textblock]) -> List[Tuple[Textblock, List[str]]]:
        textblocks_texts = [textblock.text for textblock in textblocks]
        sentence_tokenizer = SegtokSentenceSplitter()
        parsed_textblocks = [sentence_tokenizer.split(textblock) for textblock in textblocks_texts]
        [self.flair_ner_tagger.predict(sentences) for sentences in parsed_textblocks] # find entities in sentences
        assert len(textblocks) == len(parsed_textblocks)
        parsed_textblocks_entities: List[Tuple[Textblock, List[str]]] = []
        for textblock, sentencelist in list(zip(textblocks, parsed_textblocks)):
            distinct_entities: Set[str] = set()
            for sentence in sentencelist:
                ner_texts = [span.text for span in sentence.get_spans('ner')]
                distinct_entities.update(ner_texts)
            parsed_textblocks_entities.append((textblock, list(distinct_entities)))

        return parsed_textblocks_entities


    def __get_textblock_with_entities_spacy(self, textblocks: List[Textblock]) -> List[Tuple[Textblock, List[str]]]:
        # format for question generation
        parsed_textblocks_entities = []
        textblocks_texts = [textblock.text for textblock in textblocks]
        parsed_textblocks = list(self.spacy.pipe(textblocks_texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]))
        assert len(textblocks) == len(parsed_textblocks)
        for textblock, parsed_textblock in list(zip(textblocks, parsed_textblocks)):
            distinct_ents = list({ent.text for ent in parsed_textblock.ents})
            parsed_textblocks_entities.append((textblock, distinct_ents))

        return parsed_textblocks_entities


    def __generate_questions_from_textblocks_ner_highlighted_qg(self, 
                                                                textblocks: List[Textblock], 
                                                                ner_lib: NerLibrary, 
                                                                highlighting_strategy: NerHighlightingStrategy) -> List[Tuple[List[Textblock], List[str]]]:
        ner_lib_function_dict = {
            NerLibrary.spaCy: self.__get_textblock_with_entities_spacy,
            NerLibrary.flair: self.__get_textblock_with_entities_flair
        }
        ner_func = ner_lib_function_dict[ner_lib]
        hl_strategy_function_dict = {
            NerHighlightingStrategy.word: self.highlight_named_entity_in_textblock,
            NerHighlightingStrategy.sentence: self.highlight_sentence_with_named_entity_in_textblock
        }
        hl_func = hl_strategy_function_dict[highlighting_strategy]

        used_textblocks_for_generation = []
        textblocks_with_highlighted_entities = []

        for textblock, entities in ner_func(textblocks):
            if entities: # any ner texts
                for entity in entities: # foreach entity found in text generate a new question
                    highlighted_textblocks = hl_func(textblock.text, entity)
                    textblocks_with_highlighted_entities.extend(highlighted_textblocks)
                    for highlighted_text, entity in highlighted_textblocks:
                        used_textblocks_for_generation.append(
                            Textblock_with_entity.copy_from(textblock, highlighted_text, entity)
                        )
            else:
                self.logger.debug(f"No entities found in {textblock.text} textblock skipping")

        formatted_paragraphs = [f"generate question: {text} </s>" for text in textblocks_with_highlighted_entities]
        self.logger.debug(f"Generating questions from {len(formatted_paragraphs)} textblocks")
        context_dataset = ListDataset(formatted_paragraphs)
        generated_questions = []
        with tqdm(self.pipeline_german_hl_question_generation(context_dataset,
                                                              batch_size=1,
                                                              num_beams=4,
                                                              early_stopping=True,
                                                              max_length=32),
                                                              total=len(context_dataset)) as pbar:
            for out in pbar:
                pbar.set_description("Generating questions")
                generated_questions.append(out[0]['generated_text'])

        assert len(generated_questions) == len(used_textblocks_for_generation)
        return list(zip(used_textblocks_for_generation, generated_questions))


if __name__ == "__main__":
    main()