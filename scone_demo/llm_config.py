import glob
import os
import dspy.evaluate
import pandas as pd
import random

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from dataloader import load_scone

llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.settings.configure(lm=llama3)

all_train = load_scone("ScoNe/scone_nli/train")

random.seed(1)
random.shuffle(all_train)

# 200 random train, 50 random dev
train, dev = all_train[:200], all_train[200:250] 

print(len(train), len(dev))

test = load_scone("ScoNe/scone_nli/test")

# We're developing a system for the full ScoNe benchmark, but we'll
# evaluate only on one of the hardest and most informative ScoNe
# categories for now -- examples with a single negation that plays
# a crucial role in the reasoning:

test = [ex for ex in test if ex.category == "one_scoped"]

print(pd.Series([ex.answer for ex in test]).value_counts())

scone_accuracy = dspy.evaluate.metrics.answer_exact_match
evaluator = Evaluate(devset=dev, num_threads=1, display_progress=True, display_table=0)





