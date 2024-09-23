## Repo content

This Repo is intended for:


### Comparison of the 3 Qwen from version 1.5, to 2 to 2.5
- using Q4 for Qqwen 1.5 and 1.8 b
```
0.RAG_qwen2.5_autotest_CHAT.py
1.RAG_qwen2.5_autotest_CHAT.py
2.RAG_qwen2_autotest_CHAT.py
3.RAG_qwen1.5_autotest_CHAT.py
```

### Speed Comparison between LLamaCPP and Openvino with Gemma2-2B INT4/Q4
- Using gemma-2-2b-it-Q4_K_M.gguf (bartowski)
- on-gemma2-2b-it-ov-awq-int4 

We use a collection of 11 prompt for NLP tasks in batch
<br>these are the python files:
```
-a----         23/9/2024   9:15 am          46989 20.Gemma2-int4-cpp.py
-a----         23/9/2024   9:16 am          47002 21.Gemma2-int4-openvino.py
```

### LLAMA CPP MODELS
Under `models` subdirectory
```
gemma-2-2b-it-Q4_K_M.gguf
qwen1_5-1_8b-chat-q5_k_m.gguf
qwen2-1_5b-instruct-q5_k_m.gguf
qwen2.5-1.5b-instruct-q5_k_m.gguf
```

OpenVino model
Under `on-gemma2-2b-it-ov-awq-int4` subdirectory
```
config.json
generation_config.json
openvino_config.json
openvino_detokenizer.bin
openvino_detokenizer.xml
openvino_model.bin
openvino_model.xml
openvino_tokenizer.bin
openvino_tokenizer.xml
README.md
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```

