import json
import csv
from abc import ABC, abstractmethod
from typing import Optional, Any
import spacy

import pandas as pd
from datasets import Dataset, DatasetDict


class BaseText2TextDatasetTransformer(ABC):
    PREFIX = "prefix"
    INPUT_TEXT = "input_text"
    TARGET_TEXT = "target_text"
    GENERIC_TEXT_2_TEXT_FORMAT_COLUMNS = [PREFIX, INPUT_TEXT, TARGET_TEXT]


    def __init__(self,
                 tokenizer_eos: str,
                 train_path: str,
                 task_prefix: str = "",
                 validation_path: Optional[str] = None,
                 test_path: Optional[str] = None):
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self._task_prefix = task_prefix
        self._tokenizer_eos = tokenizer_eos
        self.dataset_splits = [("train", self.train_path)]
        if self.validation_path:
            self.dataset_splits.append(("validation", self.validation_path))
        if self.test_path:
            self.dataset_splits.append(("test", self.test_path))


    @property
    def task_prefix(self) -> Optional[str]:
        return self._task_prefix


    @property
    def tokenizer_eos(self) -> str:
        return self._tokenizer_eos


    @abstractmethod
    def create_dataset_dict(self) -> DatasetDict:
        """Create a dataset dict from different splits

        Returns:
            DatasetDict: Dict which contains splits for training
        """
        pass


    @abstractmethod
    def transform(self, data: Any) -> pd.DataFrame:
        """Transform a single file for the target format

        Args:
            data: object which is processed

        Returns:
            pd.DataFrame: Dataframe of transformed data
        """
        pass


# SQuAD Dataset Transformer
class SquadDatasetTransformer(BaseText2TextDatasetTransformer):
    def _germanquad_context_preprocessing(self, context: str) -> str:
        return (context.split("==\n", 1)[-1] # cut off title in context (GermanQuAD)
                       .replace("\n", " ") # collapse newlines
                       .replace("''", "'") # do not strip leading blank spaces GH-2585
        )


    def _preprocess_squad(self, examples, transformation_function, context_preprocessing_function = None):
        for article in examples:
            paragraphs = article["paragraphs"]
            for paragraph in paragraphs:
                context = context_preprocessing_function(paragraph["context"]) if context_preprocessing_function else paragraph["context"]
                yield transformation_function(paragraph, context)


    def create_dataset_dict(self) -> DatasetDict:
        dataset_dict = {}
        for split, dataset_path in self.dataset_splits:
            with open(str(dataset_path), encoding="utf-8") as f:
                squad = json.load(f)
                data = squad["data"]
                df = self.transform(data)
                dataset_dict.update({split: Dataset.from_pandas(df.copy())})
        
        return DatasetDict(dataset_dict)



    @abstractmethod
    def transform(self, data) -> pd.DataFrame:
        pass


# End 2 End Question Generation
class SquadEnd2EndQuestionGeneration(SquadDatasetTransformer):
    def _transform_squad_for_e2e_question_generation(self, paragraph, context):
        questions = [qa["question"].strip() for qa in paragraph["qas"]]
        return {
            self.PREFIX: self.task_prefix,
            self.INPUT_TEXT: context + self.tokenizer_eos,
            self.TARGET_TEXT: " <sep> ".join(questions) + self.tokenizer_eos
        }


    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(
            list(self._preprocess_squad(data, self._transform_squad_for_e2e_question_generation)))


class GermanQuadEnd2EndQuestionGeneration(SquadEnd2EndQuestionGeneration):
    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(self._preprocess_squad(data, self._transform_squad_for_e2e_question_generation, self._germanquad_context_preprocessing)))
        

# Answer Aware Question Generation
class SquadAnswerAwareQuestionGeneration(SquadDatasetTransformer):
    def _highlight_text(self,
                        text: str, 
                        context: str, 
                        highlight_token: str = "<hl>") -> str:
        test = context.replace(text, f"{highlight_token}{text}{highlight_token}")
        return test


    def _transform_squad_for_answer_aware_question_generation(self, paragraph, context):
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"]
            highlighted_context = self._highlight_text(answer, context)
            return {
                self.PREFIX: self.task_prefix, 
                self.INPUT_TEXT: highlighted_context + self._tokenizer_eos,
                self.TARGET_TEXT: question + self._tokenizer_eos
            }


    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(
            self._preprocess_squad(data, self._transform_squad_for_answer_aware_question_generation)))


class GermanQuadAnswerAwareQuestionGeneration(SquadAnswerAwareQuestionGeneration):
    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(
            self._preprocess_squad(data, 
                                   self._transform_squad_for_answer_aware_question_generation, 
                                   self._germanquad_context_preprocessing)))


# Question Answering
class SquadQuestionAnswering(SquadDatasetTransformer):
    def _transform_squad_question_answering(self, paragraph, context):
        for qa in paragraph["qas"]:
            question = qa["question"]
            return {
                self.PREFIX: self.task_prefix,
                self.INPUT_TEXT: f"context: {context} question: {question}{self.tokenizer_eos}",
                self.TARGET_TEXT: qa["answers"][0]["text"] + self.tokenizer_eos
            }


    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(self._preprocess_squad(data, self._transform_squad_question_answering)))


class GermanQuadQuestionAnswering(SquadQuestionAnswering):
    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(self._preprocess_squad(data, self._transform_squad_question_answering, self._germanquad_context_preprocessing)))


# Answer Extraction
class SquadAnswerExtraction(SquadDatasetTransformer):
    def _transform_squad_for_answer_extraction(self, paragraph, context):
        for qa in paragraph["qas"]:
            # ToDo: The sentence is highlighted in https://arxiv.org/abs/2111.06476
            answers = [answer["text"] for answer in qa["answers"]]
            return {
                self.PREFIX: self.task_prefix,
                self.INPUT_TEXT: context + self.tokenizer_eos,
                self.TARGET_TEXT: " <sep> ".join(answers) + self.tokenizer_eos
            }


    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(self._preprocess_squad(data, self._transform_squad_for_answer_extraction)))


class GermanQuadAnswerExtraction(SquadAnswerExtraction):
    def __init__(self,
                 tokenizer_eos: str,
                 train_path: str,
                 task_prefix: str = "",
                 validation_path: Optional[str] = None,
                 test_path: Optional[str] = None):
        super().__init__(tokenizer_eos, train_path, task_prefix, validation_path, test_path)
        self.senter = spacy.blank("de")
        self.senter.add_pipe("senter", source=spacy.load("de_core_news_sm"))


    def _transform_germanquad_for_answer_extraction(self, paragraph, context, highlight_token: str = "<hl>"):
        for qa in paragraph["qas"]:
            answers = [answer["text"] for answer in qa["answers"]]
            doc = self.senter(context)
            highlighted_context = ""
            for answer in answers:
                for sentence in doc.sents:
                    if answer in sentence.text:
                        highlighted_context += f"{highlight_token}{sentence.text}{highlight_token}"
                    else:
                        highlighted_context += sentence.text

            return {
                self.PREFIX: self.task_prefix,
                self.INPUT_TEXT: highlighted_context + self.tokenizer_eos,
                self.TARGET_TEXT: " <sep> ".join(answers) + self.tokenizer_eos
            }


    def transform(self, data) -> pd.DataFrame:
        return pd.DataFrame(list(self._preprocess_squad(data, self._transform_germanquad_for_answer_extraction, self._germanquad_context_preprocessing)))


# Paraphrasing
class PawsParaphrasing(BaseText2TextDatasetTransformer):
    def create_dataset_dict(self) -> DatasetDict:
        dataset_dict = {}
        for split, dataset_path in self.dataset_splits:
            paws_df = pd.read_csv(dataset_path, sep='\t', header=0, quoting=csv.QUOTE_NONE, encoding='utf-8')
            df = self.transform(paws_df)
            dataset_dict.update({split: Dataset.from_pandas(df.copy())})
        
        return DatasetDict(dataset_dict)


    def transform(self, data) -> pd.DataFrame:
        df = data[data['label'] == 1].copy()
        columns = ["id", self.INPUT_TEXT, self.TARGET_TEXT, "label"]
        df.columns = columns
        df[self.PREFIX] = self.task_prefix
        for index_label, row_series in df.iterrows():
            df.at[index_label, self.INPUT_TEXT] = str(row_series[self.INPUT_TEXT]) + self.tokenizer_eos
            df.at[index_label, self.TARGET_TEXT] = str(row_series[self.TARGET_TEXT]) + self.tokenizer_eos

        return df
