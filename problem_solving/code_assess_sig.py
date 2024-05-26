import dspy

class CodeGenAssessSignature(dspy.Signature):
    """assess the correctess and the quality of the generated code."""
    problem = dspy.InputField(desc="A problem for which we generate code")
    correct_code = dspy.InputField(desc="the provided correct code for the problem")
    assessed_code = dspy.InputField(desc="contains the generated code")
    #refined_algorithm = dspy.InputField(desc="Refined Algoritm for the given problem statement")
    assessement_question = dspy.InputField()
    assessement_answer = dspy.OutputField(desc="Yes or No")


