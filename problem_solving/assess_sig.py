import dspy

class CodeGenAssessSignature(dspy.Signature):
    """assess the correctess and the quality of the generated code."""
    assessed_code = dspy.InputField(desc="contains the generated code")
    assessed_pseudocode = dspy.InputField(desc="contains the generated pseudocode")
    assessement_question = dspy.InputField()
    assessement_answer = dspy.OutputField(desc="Yes or No")


