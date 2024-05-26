import dspy

class AlgorithmGenSignature(dspy.Signature):
    """generate an algorithm for a given problem statement."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    algorithm = dspy.OutputField(desc="Algorithm for the given problem statement")


