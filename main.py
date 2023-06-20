import os

huggingfacehub_api_token="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"

from langchain import HuggingFaceHub

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.6, "max_new_tokens":500})


from langchain import PromptTemplate, LLMChain

template = """
Eres un asistente de inteligencia artificial. El asistente da respuestas útiles,
detalladas y educadas a las preguntas del usuario.

{question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Como se hace una pizza?"

print(llm_chain.run(question))
