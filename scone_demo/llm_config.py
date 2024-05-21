import glob
import os
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



