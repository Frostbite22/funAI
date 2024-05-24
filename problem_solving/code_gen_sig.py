import dspy

class CodeGenSignature(dspy.Signature):
    """generate code for a given problem statement."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    pseudocode = dspy.InputField(desc="Pseudocode for the given problem statement")
    code = dspy.OutputField(desc="Generated python code for the given problem statement")


