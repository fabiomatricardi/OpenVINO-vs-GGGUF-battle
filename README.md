# OpenVINO-vs-GGGUF-battle
Repo with code in the prompt battle between CPU inference with OpenVINO and llamaCPP

Original Idea is the test paper from LLMWare team
- on AIPC inferences with OpenVINO were 20 to 50 time faster on CPU than on metal with llamaCPP

Decided to test it myself

### References
- [https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP](https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_3_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP)
-[ https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_GENAI&VERSION=v_2024_3_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_GENAI&VERSION=v_2024_3_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP)

### Used models
> For INT4/Q4
- https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/tree/main
- https://huggingface.co/circulus/on-qwen2-0.5b-it-int4-awq-ov
> for INT8/Q8
- https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/tree/main
- https://huggingface.co/circulus/on-qwen2-1.5b-it-int8-ov

### PIP installation
```
get the istructions
# Step 1: Create virtual environment
python -m venv openvino_env
# Step 2: Activate virtual environment
openvino_env\Scripts\activate
# Step 3: Upgrade pip to latest version
python -m pip install --upgrade pip
# Step 4: Download and install the package
pip install openvino-genai==2024.3.0
```

troubles in the next step, so torch to be installed manually from WHL file
```
https://files.pythonhosted.org/packages/5a/6a/775b93d6888c31f1f1fc457e4f5cc89f0984412d5dcdef792b8f2aa6e812/torch-2.4.1-cp311-cp311-win_amd64.whl
pip install optimum-intel[openvino]
```

for comparisons
```
pip install tiktoken
install llama-cpp-python==0.2.90
```

Ensure BLAS is not active!


