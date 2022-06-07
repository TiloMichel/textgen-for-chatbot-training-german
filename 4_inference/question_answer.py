from typing import List, Tuple, Optional
from pydantic import BaseModel
from language_tool_python import Match


class WebpageInfo(BaseModel):
    title: str
    url: str
    depth: int


class WebpageText(WebpageInfo):
    extracted_text: str
    added_position: int


class Textblock(WebpageInfo):
    text: str
    page_text_position: int
    page_added_position: int


class Textblock_with_entity(Textblock):
    entity: str


    @staticmethod
    def copy_from(textblock: Textblock, highlighted_text: str, entity: str):
        return Textblock_with_entity(
            title=textblock.title,
            url=textblock.url,
            depth= textblock.depth,
            text=highlighted_text,
            page_added_position=textblock.page_added_position, 
            page_text_position=textblock.page_text_position,
            entity=entity
        )


class QuestionsAnswerItem(WebpageInfo):
    page_added_position: int
    page_text_position: int
    question_violated_grammar_rules: List[Tuple[str, str, str, str]]
    question: str
    answer: str
    entity: Optional[str]
    # question
    question_grammar_correct: bool
    question_answerable_by_answer: bool = False
    question_useable_for_chatbot: bool = False
    # comments
    comment_question: Optional[str]
    alternative_formulation: Optional[str]


    @staticmethod
    def copy_from(textblock: Textblock_with_entity, question: str, matches: List[Match]):
        return QuestionsAnswerItem(
            page_added_position=textblock.page_added_position,
            page_text_position=textblock.page_text_position,
            title=textblock.title,
            url=textblock.url,
            depth=textblock.depth,
            question=question,
            question_violated_grammar_rules=[(
                f"rule id: {match.ruleId}",
                f"category: {match.category}",
                f"rule issue type: {match.ruleIssueType}",
                f"message: {match.message}") for match in matches],
            answer=textblock.text,
            question_grammar_correct=len(matches) <= 0,
            entity=textblock.entity if hasattr(textblock, "entity") else None
        )


class AnswerItem(WebpageInfo):
    page_added_position: int
    page_text_position: int
    answer: str
    # answer
    answer_informative: bool = False
    answer_too_short: bool = False # not defined yet, purely subjective to human annotator
    answer_at_least_one_sentence: bool = False
    # comments
    comment_answer: Optional[str]

    @staticmethod
    def copy_from(textblock: Textblock):
        return AnswerItem(
            page_added_position=textblock.page_added_position,
            page_text_position=textblock.page_text_position,
            title=textblock.title,
            url=textblock.url,
            depth=textblock.depth,
            answer=textblock.text
        )

