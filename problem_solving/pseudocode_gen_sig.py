import dspy

class PseudocodeGenSignature(dspy.Signature):
    """generate code for a given problem statement."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    pseudocode = dspy.OutputField(desc="Pseudocode for the given problem statement")


