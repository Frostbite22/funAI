import dspy

class RefinedCodeGenSignature(dspy.Signature):
    """generate a refined python code from a problem statement """
    problem = dspy.InputField(desc="A problem to solve with python code")
    algorithm = dspy.InputField(desc="Algorithm for the given problem statement")
    generated_code = dspy.InputField(desc="Generated python code for the given problem statement")
    refined_code = dspy.OutputField(desc="refined python code for the given problem statement")



