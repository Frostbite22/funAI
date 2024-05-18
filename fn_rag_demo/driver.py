#Loading:
from custom_data import CSVDataset
from dspy.evaluate.evaluate import Evaluate

from rag import RAG
import dspy
import os


data_schema = """
CREATE TABLE roles (

  role_id INT PRIMARY KEY AUTO_INCREMENT,

  title VARCHAR(100) NOT NULL

);
 
CREATE TABLE employees (

  employee_id INT PRIMARY KEY AUTO_INCREMENT,

  first_name VARCHAR(100) NOT NULL,

  last_name VARCHAR(100) NOT NULL,
  
  salary INT, 

  department_id INT,

  role_id INT,

  FOREIGN KEY (department_id) REFERENCES departments(department_id),

  FOREIGN KEY (role_id) REFERENCES roles(role_id)

);
 
CREATE TABLE projects (

  project_id INT PRIMARY KEY AUTO_INCREMENT,

  project_name VARCHAR(255) NOT NULL,

  start_date DATE,

  end_date DATE,

  department_id INT,

  FOREIGN KEY (department_id) REFERENCES departments(department_id)

);
 
CREATE TABLE departments (

  department_id INT PRIMARY KEY AUTO_INCREMENT,

  name VARCHAR(100) NOT NULL,

  location VARCHAR(255)

);
 
 
CREATE TABLE timesheets (

  timesheet_id INT PRIMARY KEY AUTO_INCREMENT,

  employee_id INT NOT NULL,

  project_id INT NOT NULL,

  date_worked DATE NOT NULL,

  hours_worked DECIMAL(5, 2) NOT NULL,

  FOREIGN KEY (employee_id) REFERENCES employees(employee_id),

  FOREIGN KEY (project_id) REFERENCES projects(project_id)

);
"""
# Ask any question you like to this simple RAG program.
question = "what are the functions and their parameters ?"

user_story = "As a user, I want to get employees by department name "


# Language model
llama3 = dspy.GROQ(model='llama3-70b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.configure(lm=llama3)

#Retrieval model
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=llama3,rm=colbertv2_wiki17_abstracts)


rag = RAG()
rag.load('fn_rag_demo/llama3_70b_v0.json')

pred = rag(user_story=user_story,data_schema=data_schema,question=question)

# Print the contexts and the answer.
print(f"User Story: {pred}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")


### Evaluate the model on a custom dataset : timedout 
# Loading the data
# dataset = CSVDataset('rag_data.xlsx')

# trainset = [x.with_inputs('data_schema','user_story','question') for x in dataset.train]
# devset = [x.with_inputs('data_schema','user_story','question') for x in dataset.dev]


# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
# evaluate_on_custom_dataset = Evaluate(devset=devset, num_threads=1, display_progress=False, display_table=5)

# Evaluate the `compiled_rag` program with the `answer_exact_match` metric.

# def validate(example, pred, trace=None):
#     return example.answer.lower() == pred.answer.lower()

# print(evaluate_on_custom_dataset(rag, metric=validate))





