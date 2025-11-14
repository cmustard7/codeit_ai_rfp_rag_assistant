from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-5-mini', temperature=0.0)
classifier_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
judge_llm = ChatOpenAI(model='gpt-5-mini', temperature=0.0)