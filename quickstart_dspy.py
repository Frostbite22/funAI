import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import os

# Set up the LM
llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.settings.configure(lm=llama3)

# Load math questions from the GSM8K dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

class CoT(dspy.Module):
    # This is our Signature "question -> answer"
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
    

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)


from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
#evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
#evaluate(optimized_cot)

print(optimized_cot(question="My sister bought 3 apples and 4 oranges. I bought half the number of oranges she bought and twice the number of apples she bought. How many fruits did I buy?"))