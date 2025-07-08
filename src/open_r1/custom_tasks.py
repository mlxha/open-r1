# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
import lighteval.tasks.default_prompts as prompt


latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    query = query_template.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"])

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )

def medqa_gen_prompt_fn(line, task_name: str = None):
    """Custom prompt function for MedQA generative evaluation with thinking template"""
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion: {question}\n\n{options}\n"
    
    # Format the options from MedQA structure
    options_text = "".join([f"{option['key']}) {option['value']}\n" for option in line["options"]])
    
    query = query_template.format(
        question=line['question'],
        options=options_text.strip()
    )
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=[opt["key"] for opt in line["options"]],
        gold_index=[opt["key"] for opt in line["options"]].index(line["answer_idx"]),
        instruction=query,
    )

def medmcqa_instruct_prompt_fn(line, task_name: str = None):
    """Custom prompt function for MedMCQA generative evaluation with thinking template"""
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion: {question}\n\nA) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\n"
    
    query = query_template.format(
        question=line['question'],
        opa=line['opa'],
        opb=line['opb'],
        opc=line['opc'],
        opd=line['opd']
    )
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=line["cop"] - 1,  # cop is 1-indexed, convert to 0-indexed
        instruction=query,
    )

def medxpertqa_prompt_fn(line, task_name: str = None):
    """Custom prompt function for MedXpertQA evaluation with thinking template"""
    query_template = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFGHIJ. Think step by step before answering.\n\n{question}"
    
    query = query_template.format(question=line['question'])
    
    # Extract all possible choice letters from the options
    choices = list(line["options"].keys())  # Should be ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    
    # Find the index of the correct answer
    gold_index = choices.index(line["label"])
    
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=query,
    )

# Define tasks
medqa_gen = LightevalTaskConfig(
    name="medqa_gen",
    suite=["custom"],
    prompt_function=medqa_gen_prompt_fn,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[
        Metrics.gpqa_instruct_pass_at_1_1n,
        Metrics.gpqa_instruct_pass_at_1_4n,
    ],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=2,
)

medmcqa_gen = LightevalTaskConfig(
    name="medmcqa_gen",
    suite=["custom"],
    prompt_function=medmcqa_instruct_prompt_fn,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[
        Metrics.gpqa_instruct_pass_at_1_1n,
        Metrics.gpqa_instruct_pass_at_1_4n,
    ],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=2,
)

medxpertqa_gen = LightevalTaskConfig(
    name="medxpertqa_gen",
    suite=["custom"],
    prompt_function=medxpertqa_prompt_fn,
    hf_repo="TsinghuaC3I/MedXpertQA",
    hf_subset="Text",
    hf_avail_splits=["dev", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8192,
    metric=[
        Metrics.gpqa_instruct_pass_at_1_1n,
        Metrics.gpqa_instruct_pass_at_1_4n,
    ],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=2,
)

pubmedqa = LightevalTaskConfig(
    name="pubmedqa",
    suite=["custom"],
    prompt_function=prompt.pubmed_qa,
    hf_repo="pubmed_qa",
    hf_subset="pqa_labeled",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=1,
)

medmcqa = LightevalTaskConfig(
    name="medmcqa",
    suite=["custom"],
    prompt_function=prompt.med_mcqa,
    hf_repo="lighteval/med_mcqa",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=1,
)

medqa = LightevalTaskConfig(
    name="medqa",
    suite=["custom"],
    prompt_function=prompt.med_qa,
    hf_repo="bigbio/med_qa",
    hf_subset="med_qa_en_source",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=1,
)

aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[expr_gold_metric],
    version=1,
)
math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)
gpqa_diamond = LightevalTaskConfig(
    name="gpqa:diamond",
    suite=["custom"],
    prompt_function=prompt.gpqa_instruct,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,  # needed for reasoning models like R1
    metric=[gpqa_metric],
    stop_sequence=[],  # no stop sequence, will use eos token
    trust_dataset=True,
    version=1,
)


# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(medqa_gen)
TASKS_TABLE.append(medmcqa_gen)
TASKS_TABLE.append(medxpertqa_gen)
TASKS_TABLE.append(pubmedqa)
TASKS_TABLE.append(medmcqa)
TASKS_TABLE.append(medqa)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))