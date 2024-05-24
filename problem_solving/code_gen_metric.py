import dspy

from assess_sig import CodeGenAssessSignature 

llama3 = dspy.OllamaLocal(model='llama3')

def metric(example,pred,trace=None):
    problem, correct_code, generated_code, psdcd = example.problem, example.python, pred.generated_code, pred.psdcd

    correct = f"the generated code : {generated_code.code} \n should be considered correct and of high quality if it has the same complexity of {correct_code} or lower and implements the correct logic with the help of {psdcd.pseudocode} for the given problem {problem}.\nDoes the generated code meet this criteria? Answer Yes or No."
    print("Problem:",problem,"\n")
    print("Pseudocode:",psdcd.pseudocode,"\n")
    print("Generated code:",generated_code.code,"\n")
    print("Correct code:",correct_code,"\n")

    with dspy.context(lm=llama3):
        correct = dspy.Predict(CodeGenAssessSignature)(assessed_code=generated_code.code,assessement_question=correct,assessed_pseudocode=psdcd.pseudocode)

    print("correct:",correct,"\n")
    score = 1 if 'yes' in correct.assessement_answer.lower() else 0

    if trace is not None: return score >= 0
    return score 



