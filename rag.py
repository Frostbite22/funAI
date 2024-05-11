import dspy

from signature import GenerateAnswer

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, user_story,data_schema,question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context,question=question,user_story=user_story,data_schema=data_schema)
        return dspy.Prediction(context=context,answer=prediction.answer)

