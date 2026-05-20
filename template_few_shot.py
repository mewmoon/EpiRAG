from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms.tongyi import Tongyi

example_prompt = PromptTemplate.from_template("单词：{word}, 反义词{antonym}")
example_data = [{"word": "大", "antonym": "small"}, {"word": "上", "antonym": "down"}]
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=example_data,
    prefix="给出给定词的反义词，有如下示例：",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=["input_word"],
)
prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()

model = Tongyi(model="qwen-max")
res = model.invoke(prompt_text)
print(res)
