import dspy

class AlgorithmGenAssessSignature(dspy.Signature):
    """assess the quality the quality of the algorithm generated."""
    problem = dspy.InputField(desc="A problem for which we generate pseudocode")
    correct_code = dspy.InputField(desc="the provided correct code for the problem with solution description")
    assessed_algorithm = dspy.InputField(desc="contains the generated algorithm")
    assessement_question = dspy.InputField()
    assessement_answer = dspy.OutputField(desc="Yes or No")


