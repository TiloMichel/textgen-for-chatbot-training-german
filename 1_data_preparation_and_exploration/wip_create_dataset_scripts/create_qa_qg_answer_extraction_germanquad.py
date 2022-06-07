import argparse
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets
from Text2TextDatasetTransformers import (
    GermanQuadAnswerAwareQuestionGeneration, 
    GermanQuadQuestionAnswering, 
    GermanQuadAnswerExtraction)


def main(args):
    dataset_path = Path(args.dir)
    germanquad_path = dataset_path / "germanquad"
    train_file = str(germanquad_path / "GermanQuAD-train.json")
    validation_file = str(germanquad_path / "GermanQuAD-test.json")
    eos_token = "</s>"
    answer_aware_germanquad_transformator = GermanQuadAnswerAwareQuestionGeneration(
        task_prefix="generate question: ",
        tokenizer_eos=eos_token,
        train_path=train_file,
        validation_path=validation_file)
    answer_aware_qg_dict = answer_aware_germanquad_transformator.create_dataset_dict()
    print(answer_aware_qg_dict['train'][0])
    question_answering_germanquad_transformator = GermanQuadQuestionAnswering(
        tokenizer_eos=eos_token,
        train_path=train_file,
        validation_path=validation_file)
    question_answering_dict = question_answering_germanquad_transformator.create_dataset_dict()
    print(question_answering_dict['train'][0])
    answer_extraction_transformator = GermanQuadAnswerExtraction(
        task_prefix="extract answer: ",
        tokenizer_eos=eos_token,
        train_path=train_file,
        validation_path=validation_file)
    answer_extraction_dict = answer_extraction_transformator.create_dataset_dict()
    print(answer_extraction_dict['train'][0])
    concatenated_splits = {}
    for split in answer_aware_germanquad_transformator.dataset_splits:
        concatenated_splits.update(
            {split[0]: concatenate_datasets([answer_aware_qg_dict[split[0]], question_answering_dict[split[0]], answer_extraction_dict[split[0]]])})

    qa_qg_aa_dataset_dict = DatasetDict(concatenated_splits)
    qa_qg_aa_dataset_dict.save_to_disk(str(dataset_path / "qa_qg_aa_germanquad"))
    qa_qg_aa_dataset_dict.load_from_disk(str(dataset_path / "qa_qg_aa_germanquad")) # check if it works


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", 
        type=str, 
        default="", 
        help="Path to the datasets directory.")
    args = parser.parse_args()
    main(args)