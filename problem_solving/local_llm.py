import dspy

from dataloader import PrbSolvDataset 
from dspy.evaluate import Evaluate

from code_gen_metric import metric
from module import ProblemSolvingModule

llama3 = lm = dspy.OllamaLocal(model='llama3')
dspy.configure(lm=llama3)

dataset = PrbSolvDataset('problem_solving/dataset.xlsx')
trainset = [x.with_inputs('problem') for x in dataset.train]
devset = [x.with_inputs('problem') for x in dataset.dev]

pot_zeroshot = ProblemSolvingModule()
# result = pot_zeroshot("problem statement : I want to calculate the sum of two numbers a and b. \n Can you generate python code for the given problem statement ?")
# print(result)

evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=0)

print(evaluator(pot_zeroshot, metric=metric))
