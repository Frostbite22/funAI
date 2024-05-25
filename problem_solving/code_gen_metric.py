import dspy

from code_assess_sig import CodeGenAssessSignature
from pseudocode_assess_sig import PseudocodeGenAssessSignature 

llama3 = dspy.OllamaLocal(model='llama3')

def metric(example,pred,trace=None):
    problem, correct_code, generated_code, psdcd = example.problem, example.python, pred.generated_code, pred.psdcd

    correct_assessed_code = "Is the generated code correct by reference to the correct code and the problem statement? Answer Yes or No."

    correct_assessed_pseudocode = "Is the generated pseudocode correct by reference to the problem statement and the comments in the correct code? Answer Yes or No."
    print("Problem:",problem,"\n")
    print("Pseudocode:",psdcd.pseudocode,"\n")
    print("Generated code:",generated_code.code,"\n")
    print("Correct code:",correct_code,"\n")

    with dspy.context(lm=llama3):
        correct_assessed_pseudocode = dspy.Predict(PseudocodeGenAssessSignature)(assessement_question=correct_assessed_pseudocode,assessed_pseudocode=psdcd.pseudocode,correct_code=correct_code,problem=problem)
        correct_assessed_code = dspy.Predict(CodeGenAssessSignature)(assessed_code=generated_code.code,assessement_question=correct_assessed_code,assessed_pseudocode=psdcd.pseudocode,correct_code=correct_code,problem=problem)

    print("assessed pseudocode:",correct_assessed_pseudocode,"\n")
    print("assessed code:",correct_assessed_code,"\n")
    
    score = 1 if 'yes' in correct_assessed_code.assessement_answer.lower() and 'yes' in correct_assessed_pseudocode.assessement_answer.lower() else 0

    if trace is not None: return score >= 0
    return score 



