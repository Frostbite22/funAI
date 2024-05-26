import dspy

class CodeGenSignature(dspy.Signature):
    """generate python code for a given problem statement based on an algorithm."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    algorithm = dspy.InputField(desc="Algorithm for the given problem statement")
    generated_code = dspy.OutputField(desc="Generated python code for the given problem statement based on the Algorithm")


