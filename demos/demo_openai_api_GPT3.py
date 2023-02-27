"""
    本例程演示了如何调用openai的api实现对话。
"""
import openai

# Set your API key
openai.api_key = "your_api_key"
# Use the GPT-3 model
completion = openai.Completion.create(
    engine="text-davinci-003",
    prompt="""
观察以下计算结果，回答问题。尽可能使用自然、亲切的语言，使用以下知识而不是你自己的知识，但是你可以尝试着对其中的专业词语附带进行通俗的解释。最后，要时刻注意安慰患者的心情，给予患者鼓励。
4【-】肺泡蛋白质沉积症【-】症状【-】【?】 毓 卓 0.99811 肺泡 炎症 0.95827 乏力 0.93251 呼吸困难 0.92865  呼 吸 衰 竭 0.91608紫 绀 0.91144 肺泡 浸润 0.90916 肺泡 灌 洗 液 可见... 0.90303
肌肉 酸痛 0.89814 胸痛 0.89472
现在，有悲伤的患者想知道肺泡蛋白质沉积症的典型表现有哪些？”，请鼓励他，并结合上述知识回答问题。
""",
    max_tokens=1024,
    temperature=0.5
)
# Print the generated text
print(completion.choices[0].text)

