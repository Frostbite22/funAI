import dspy

from code_assess_sig import CodeGenAssessSignature
from algorithm_assess_sig import AlgorithmGenAssessSignature 

llama3 = dspy.OllamaLocal(model='llama3')

def metric(example,pred,trace=None):
    problem, correct_code, code, algo = example.problem, example.python, pred.code, pred.algo

    correct_assessed_code = "Is the generated code correct by reference to the correct code and the problem statement? Answer Yes or No."

    correct_assessed_algorithm = "Is the generated algorithm correct by reference to the problem statement and the comments in the correct code? Answer Yes or No."
    print("Problem:",problem,"\n")
    print("Refined Algorithm:",algo.refined_algorithm,"\n")
    print("Refined code:",code.refined_code,"\n")
    print("Correct code:",correct_code,"\n")

    with dspy.context(lm=llama3):
        correct_assessed_algorithm = dspy.Predict(AlgorithmGenAssessSignature)(assessement_question=correct_assessed_algorithm,assessed_algorithm=algo.refined_algorithm,correct_code=correct_code,problem=problem)
        correct_assessed_code = dspy.Predict(CodeGenAssessSignature)(assessed_code=code.refined_code,assessement_question=correct_assessed_code,correct_code=correct_code,problem=problem)

    print("assessed algorithm:",correct_assessed_algorithm,"\n")
    print("assessed code:",correct_assessed_code,"\n")
    print("##############################################")

    score = 1 if 'yes' in correct_assessed_code.assessement_answer.lower() and 'yes' in correct_assessed_algorithm.assessement_answer.lower() else 0

    if trace is not None: return score >= 0
    return score 



