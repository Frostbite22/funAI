import dspy

class ProblemSolvingSignature(dspy.Signature):
    """generate code for a given problem statement."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    code = dspy.OutputField(desc="Generated python code for the given problem statement")


