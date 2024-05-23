import glob
import os
import dspy.evaluate
import pandas as pd
import random

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from dataloader import load_scone
from module import ScoNeCoT

llama3 = lm = dspy.OllamaLocal(model='llama3')
dspy.settings.configure(lm=llama3)

RUN_FROM_SCRATCH = False

all_train = load_scone("ScoNe/scone_nli/train")

random.seed(1)
random.shuffle(all_train)

# 200 random train, 50 random dev
train, dev = all_train[:200], all_train[200:250] 

# print(len(train), len(dev))

test = load_scone("ScoNe/scone_nli/test")

# We're developing a system for the full ScoNe benchmark, but we'll
# evaluate only on one of the hardest and most informative ScoNe
# categories for now -- examples with a single negation that plays
# a crucial role in the reasoning:

test = [ex for ex in test if ex.category == "one_scoped"]

# print(pd.Series([ex.answer for ex in test]).value_counts())

scone_accuracy = dspy.evaluate.metrics.answer_exact_match
evaluator = Evaluate(devset=dev, num_threads=1, display_progress=True, display_table=0)

cot_zeroshot = ScoNeCoT()

print(evaluator(cot_zeroshot, metric=scone_accuracy))

# Optimized few-shot with bootstrapped demonstrations
bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    metric=scone_accuracy,
    num_candidate_programs=10,
    num_threads=8,
    teacher_settings=dict(lm=llama3)
    )

if RUN_FROM_SCRATCH:
    cot_fewshot = bootstrap_optimizer.compile(ScoNeCoT(), trainset=train, valset=dev)
    cot_fewshot.save("scone_demo/compiled_cot_fewshot.json")
else:
    cot_fewshot = ScoNeCoT()
    cot_fewshot.load("scone_demo/compiled_cot_fewshot.json")

print(evaluator(cot_fewshot, metric=scone_accuracy))
# Exampe prompt with prediction 

print(llama3.inspect_history(n=1))
