import glob
import os
import pandas as pd
import random

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.settings.configure(lm=llama3)


