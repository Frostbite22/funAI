import dspy

from dataloader import PrbSolvDataset 
from dspy.evaluate import Evaluate

from code_gen_metric import metric
from module import ProblemSolvingModule

from dspy.teleprompt import BootstrapFewShotWithRandomSearch


llama3 = lm = dspy.OllamaLocal(model='llama3')
dspy.configure(lm=llama3)

dataset = PrbSolvDataset('problem_solving/dataset.xlsx')
trainset = [x.with_inputs('problem') for x in dataset.train]
devset = [x.with_inputs('problem') for x in dataset.dev]

pot_zeroshot = ProblemSolvingModule()
# result = pot_zeroshot("problem statement : I want to calculate the sum of two numbers a and b. \n Can you generate python code for the given problem statement ?")
# print(result)

evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=0)

# print(evaluator(pot_zeroshot, metric=metric))

bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    num_candidate_programs=1,
    num_threads=4,
    metric=metric,
    teacher_settings=dict(lm=llama3))

pot_fewshot = bootstrap_optimizer.compile(pot_zeroshot, trainset=trainset, valset=devset)

pot_fewshot.save("problem_solving/compiled_pot_fewshot.json")

print(evaluator(pot_fewshot, metric=metric))

print(llama3.inspect_history(n=1))



