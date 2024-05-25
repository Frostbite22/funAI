import dspy

from module import ProblemSolvingModule
import time

problem = "As a user, I want to modify a Binary Search Tree so that all greater values in the given BST are added to every node. Can you generate python code for the given problem statement ?"

start = time.time()
# Language model
llama3 = dspy.OllamaLocal(model='llama3')
dspy.configure(lm=llama3)



pot_zeroshot = ProblemSolvingModule()
pot_zeroshot.load('problem_solving\compiled_pot_fewshot.json')

pred = pot_zeroshot(problem=problem)

# Print the contexts and the answer.
print(f"problem: {problem}","\n")
print(f"Predicted pseudocode : {pred.psdcd.pseudocode}","\n")
print("##############################################")
print(f"Predicted code: {pred.generated_code.code}","\n")

end = time.time()
print("Time taken:",end-start)

# print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

