import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with exact short answers."""
    data_schema = dspy.InputField(desc="contains relevant table schemas and their attributes that helps in answering the question")
    context = dspy.InputField(desc="is a user story and the context in which the question is asked")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="must return functions and their parameters")

