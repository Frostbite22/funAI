import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with exact short answers."""
    data_schema = dspy.InputField(desc="contains relevant table schemas and their attributes that helps in answering the question")
    user_story = dspy.InputField(desc="is a user story that describes the context")
    question = dspy.InputField(desc="is the question asked by the user")
    answer = dspy.OutputField(desc="must return functions and their parameters")

