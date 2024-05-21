import dspy

from scone_demo.signature import ScoNeSignature 

class ScoNeCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(ScoNeSignature)

    def forward(self,context,question):
        return self.generate_answer(context=context,question=question)