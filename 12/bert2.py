from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 根据学号选取的句子开头
prompt = "有一天，城市突然停电了"

# 加载预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"  # Hugging Face上的中文GPT模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 将模型设置为评估模式
model.eval()

# 文本生成配置
generation_config = {
    "max_length": 100,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "num_return_sequences": 1
}

# 生成续写文本
def generate_continuation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_config
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 执行生成
generated_text = generate_continuation(prompt)

# 输出结果
print("=== 续写结果 ===")
print(generated_text)
