import openai
import random
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
os.environ["OPENAI_API_BASE"] = "https://openai.api2d.net/v1"
os.environ["OPENAI_API_KEY"] = "fk193574-5HdpyDIg" ## 换成自己的key
from agent_prompts import *
class llmagent:
    def __init__(self,):
        self.llm = OpenAI(model_name="gpt-3.5-turbo")
        template = """{question}\n\n"""
        self.prompt = PromptTemplate(template=template, input_variables=["question"])

    def generate_chat_completion(self, question):
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        return llm_chain.run(question)



    def generate_decision(self,state):
        # 创建游戏提示信息，包括游戏规则、其他玩家的出牌信息和你的手牌
        id_list=[0,1]
        get_opposite_id = lambda id_list, input_id: id_list[1] if input_id == id_list[0] else id_list[0] if input_id == id_list[1] else None
        rule = 'assume you are a driving assistant'
        prompt=f"""
                You, the 'ego' car, are now driving a car on a merge section. You have already drive for {self.sce.frame} seconds.
                The decision you made LAST time step was `{last_step_action}`. Your explanation was `{last_step_explanation}`. 
                Here is the current scenario: \n ```json\n{self.sce.export2json()}\n```\n. 
                Please make decision for the `ego` car. You have to describe the state of the `ego`, then analyze the possible actions, and finally output your decision. 

                There are several rules you need to follow when you drive on a highway:
                {TRAFFIC_RULES}

                Here are your attentions points:
                {DECISION_CAUTIONS}
                
                Let's think step by step. Once you made a final decision, output it in the following format: \n
                ```
                Final Answer: 
                    "decision":{{"ego car's decision, ONE of the available actions"}},
                    "expalanations":{{"your explaination about your decision, described your suggestions to the driver"}}
                ``` \n
                """,
        # 调用AI模型生成决策
        # print('input prompt',input_prompt)
        

        output= self.generate_chat_completion(prompt) ## 这里可能生成一些不符合要求的输出，可以考虑参考drivelikehuman的处理方式
        print('AI Output:', output)
        return output                 ## 这里输出了包含解释以及llm的决策


