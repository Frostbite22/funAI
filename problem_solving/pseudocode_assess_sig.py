import dspy

class PseudocodeGenAssessSignature(dspy.Signature):
    """assess the quality the quality of the psuedocode generated."""
    problem = dspy.InputField(desc="A problem for which we generate pseudocode")
    correct_code = dspy.InputField(desc="the provided correct code for the problem with solution description")
    assessed_pseudocode = dspy.InputField(desc="contains the generated pseudocode")
    assessement_question = dspy.InputField()
    assessement_answer = dspy.OutputField(desc="Yes or No")


