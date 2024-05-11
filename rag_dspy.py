import groq
import dspy
import os
from custom_data import CSVDataset

# Language model
llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.configure(lm=llama3)

#Retrieval model
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=llama3,rm=colbertv2_wiki17_abstracts)

# Loading the data
dataset = CSVDataset('rag_data.xlsx')

print(len(dataset.train), len(dataset.dev))



