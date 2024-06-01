import dspy

from code_gen_sig import CodeGenSignature
from refined_code_gen_sig import RefinedCodeGenSignature
from algorithm_gen_sig import AlgorithmGenSignature
from refined_algorithm_sig import RefinedAlgorithmSignature

class ProblemSolvingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_algorithm = dspy.ChainOfThought(AlgorithmGenSignature)
        self.refine_algorithm = dspy.ChainOfThought(RefinedAlgorithmSignature)
        self.generate_code = dspy.ChainOfThought(CodeGenSignature)
        self.refine_code = dspy.ChainOfThought(RefinedCodeGenSignature)


    def forward(self,problem):
        algo = self.generate_algorithm(problem=problem)
        refined_algo = self.refine_algorithm(problem=problem,algorithm=algo.algorithm)
        generated_code = self.generate_code(problem=problem.replace("an algorithm","a python code"),algorithm=refined_algo.refined_algorithm)
        refined_code = self.refine_code(problem=problem.replace("an algorithm","a python code"),algorithm=refined_algo.refined_algorithm,generated_code=generated_code.generated_code)

        return dspy.Prediction(code=refined_code,algo=refined_algo)

