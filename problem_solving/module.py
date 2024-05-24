import dspy

from prb_solv_sig import ProblemSolvingSignature

class ProblemSolvingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_code = dspy.ChainOfThought(ProblemSolvingSignature)

    def forward(self,problem):
        generated_code = self.generate_code(problem=problem)
        return dspy.Prediction(generated_code=generated_code)

