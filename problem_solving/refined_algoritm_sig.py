import dspy

class RefinedAlgorithmSignature(dspy.Signature):
    """generate a refined algorithm from a given problem statement and a previously generated algoritm."""
    problem = dspy.InputField(desc="A problem to solve with python code")
    algorithm = dspy.InputField(desc="Algorithm for the given problem statement")
    refined_algorithm = dspy.OutputField(desc="Refined Algoritm for the given problem statement")



