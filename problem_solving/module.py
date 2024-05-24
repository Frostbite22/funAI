import dspy

from code_gen_sig import CodeGenSignature
from pseudocode_gen_sig import PseudocodeGenSignature

class ProblemSolvingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_pseudocode = dspy.ChainOfThought(PseudocodeGenSignature)
        self.generate_code = dspy.ChainOfThought(CodeGenSignature)

    def forward(self,problem):
        psdcd = self.generate_pseudocode(problem=problem)
        generated_code = self.generate_code(problem=problem,pseudocode=psdcd.pseudocode)

        return dspy.Prediction(generated_code=generated_code,psdcd=psdcd)

