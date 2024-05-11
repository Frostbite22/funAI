from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import dspy
import os

# Set up the LM
llama3 = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ.get("GROQ_API_KEY"))
dspy.settings.configure(lm=llama3)

host = 'localhost'
port = '3306'
username = 'root'
password = ''
database_schema = 'employee_management'
mysql_uri = f"mysql+phpmyadmin://{username}:{password}@{host}:{port}/{database_schema}"
TABLES = ['employees','roles','timesheets','projects','departments']

db = SQLDatabase.from_uri(mysql_uri, include_tables=TABLES,sample_rows_in_table_info=2)

db_chain = SQLDatabaseChain.from_llm(llama3, db, verbose=True)
