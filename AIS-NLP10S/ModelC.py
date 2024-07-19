###########################################################################################################
import sys, os
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    os.system("pip install -qqq ctransformers torch")
    
def get_LLM(temperature=0.2, top_k = 40, top_p = 0.95, num_ctx = 1024):
    if not IN_COLAB:
        raise Exception("Not running in Colab!")
    
    from ctransformers import AutoModelForCausalLM
    import torch

    """1(c). Initialize settings"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if device != "cuda":
        raise Exception("Colab GPU not available!")
    
    dtype = torch.float16 if device == "cuda" else torch.float32

    """2(a). Defining Model location"""

    # select model from https://huggingface.co/models
    # huggingface path (can also use locally downloaded model-path)
    model_name = "TheBloke/zephyr-7B-beta-GGUF"
    model_file = "zephyr-7b-beta.Q4_K_M.gguf"
    model_type="mistral"

    """2(b). Instantiating Model object"""

    # download model from huggingface
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        model_file=model_file,
        model_type=model_type,
        gpu_layers=50,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        context_length=num_ctx,
        )
    print('Loaded Zephyr Model\n')
    return llm
    
###########################################################################################################
# Example usage
if __name__ == "__main__":
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    import os
    os.system("clear")
    
    llm = get_ColabLLM()
    response = llm.invoke("Why is the sky blue?").content
    print(response)

    
