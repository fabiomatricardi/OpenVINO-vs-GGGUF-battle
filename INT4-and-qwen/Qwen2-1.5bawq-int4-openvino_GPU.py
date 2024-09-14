#from transformers import AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import datetime
import tiktoken

encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))

def countTokens(text):
    encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))
    numoftokens = len(encoding.encode(text))
    return numoftokens

model_id = "on-qwen2-1.5b-it-int4-awq-ov"
#model = AutoModelForCausalLM.from_pretrained(model_id)
# here the model was already exported so no need to set export=True
print('Loading the model and pipeline...')
start = datetime.datetime.now()
model = OVModelForCausalLM.from_pretrained(model_id).to("GPU")
#model.to("GPU")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                    max_length = 1024, 
                                    do_sample=True,
                                    temperature=0.25,
                                    top_p=0.5,
                                    repetition_penalty = 1.187,
                                    )
print(f'Model on-qwen2-1.5b-it-int4-awq-ov OpenVino and pipeline loaded in {datetime.datetime.now() - start}')
print('start inference...')
start = datetime.datetime.now()
chat = [
    {"role": "user", "content": "Explain in details what is Science."}
]

# FORMATTED TEXT FROM THE TOKENIZER TEMPLATE
#formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#print("Formatted chat:\n", formatted_chat)
print('---')
print(f"USER> {chat[0]['content']}")
results = pipe(chat,)
delta = datetime.datetime.now() - start
totalseconds = delta.total_seconds()
print(f'Inference tine: {delta}')
print('---')
output = results[0]['generated_text'][1]['content']
print(f'ASSISTANT> {output}')
prompttokens = countTokens(chat[0]['content'])
assistanttokens = countTokens(output)
totaltokens = prompttokens + assistanttokens
speed = totaltokens/totalseconds
genspeed = assistanttokens/totalseconds
print('---')
print(f"Prompt Tokens: {prompttokens}")
print(f"Output Tokens: {assistanttokens}")
print(f"TOTAL Tokens: {totaltokens}")
print('---')
print(f'>>>Inference speed: {speed:.3f}  t/s')
print(f'>>>Generation speed: {genspeed:.3f}  t/s')
print('\n\n')
start = datetime.datetime.now()
chat = [
    {"role": "user", "content": "Explain in details the differences between Machine Learning and Generative AI."}
]

# FORMATTED TEXT FROM THE TOKENIZER TEMPLATE
#formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#print("Formatted chat:\n", formatted_chat)
print('---')
print(f"USER> {chat[0]['content']}")
results = pipe(chat,)
delta = datetime.datetime.now() - start
totalseconds = delta.total_seconds()
print(f'Inference tine: {delta}')
print('---')
output = results[0]['generated_text'][1]['content']
print(f'ASSISTANT> {output}')
prompttokens = countTokens(chat[0]['content'])
assistanttokens = countTokens(output)
totaltokens = prompttokens + assistanttokens
speed = totaltokens/totalseconds
genspeed = assistanttokens/totalseconds
print('---')
print(f"Prompt Tokens: {prompttokens}")
print(f"Output Tokens: {assistanttokens}")
print(f"TOTAL Tokens: {totaltokens}")
print('---')
print(f'>>>Inference speed: {speed:.3f}  t/s')
print(f'>>>Generation speed: {genspeed:.3f}  t/s')
print('\n\n')
myprompt = """explain why it is crucial for teachers to learn how to use generative AI for their job and for the future of education.
Include relevant learning path for teachers and educators."""
templateREFLECTION = f"""You are an AI assistant designed to provide detailed, step-by-step responses. Your outputs should follow this structure:
1. Begin with a <thinking> section.
2. Inside the thinking section:
   a. Briefly analyze the question and outline your approach.
   b. Present a clear plan of steps to solve the problem.
   c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
3. Include a <reflection> section for each idea where you:
   a. Review your reasoning.
   b. Check for potential errors or oversights.
   c. Confirm or adjust your conclusion if necessary.
4. Be sure to close all reflection sections.
5. Close the thinking section with </thinking>.
6. Provide your final answer in an <output> section.
Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. Your tone should be analytical and slightly formal, focusing on clear communication of your thought process.
Remember: Both <thinking> and <reflection> MUST be tags and must be closed at their conclusion
Make sure all <tags> are on separate lines with no other text. Do not include other text on a line containing a tag.

user question: {myprompt}
"""
start = datetime.datetime.now()
chat = [
    {"role": "user", "content": templateREFLECTION}
]
print('---')
print(f"USER> {chat[0]['content']}")
results = pipe(chat,)
delta = datetime.datetime.now() - start
totalseconds = delta.total_seconds()
print(f'Inference tine: {delta}')
print('---')
output = results[0]['generated_text'][1]['content']
print(f'ASSISTANT> {output}')
prompttokens = countTokens(chat[0]['content'])
assistanttokens = countTokens(output)
totaltokens = prompttokens + assistanttokens
speed = totaltokens/totalseconds
genspeed = assistanttokens/totalseconds
print('---')
print(f"Prompt Tokens: {prompttokens}")
print(f"Output Tokens: {assistanttokens}")
print(f"TOTAL Tokens: {totaltokens}")
print('---')
print(f'>>>Inference speed: {speed:.3f}  t/s')
print(f'>>>Generation speed: {genspeed:.3f}  t/s')