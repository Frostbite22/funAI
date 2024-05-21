import dspy 

class ScoNeSignature(dspy.Signature):
    ("""You are given some context (a premise) and a question."""
     """You must indicate with Yes/No answer whether we can logically conclude the hypothesis from the premise.""")
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Yes or No")