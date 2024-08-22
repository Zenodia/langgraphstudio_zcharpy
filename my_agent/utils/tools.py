# test run and see that you can genreate a respond successfully 
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain import prompts, chat_models, hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import requests
import base64, io
from PIL import Image
import requests, json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import LLM

nvapi_key = os.environ["NVIDIA_API_KEY"]
llm=ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

def llm_rewrite_to_image_prompts(user_query):
    prompt = prompts.ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following user query into a very short, one-sentence theme for image generation, MUST follow this format : A 3D full body anime image with white background"
            ),
            ("user", "{input}"),
        ]
    )
    chain = ( prompt    | llm   | StrOutputParser() )
    out= chain.invoke({"input":user_query})
    #print(type(out))
    return out

def llm_rewrite_to_image_name(user_query):
    img_name = "output.jpg"
    return img_name



def generate_image( prompt :str) -> str :
    """
    generate image from text
    Args:
        prompt: input text
    """
    ## re-writing the input promotion title in to appropriate image_gen prompt 
    gen_prompt=llm_rewrite_to_image_prompts(prompt)
    print("start generating image with llm re-write prompt:", gen_prompt)
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {nvapi_key}",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{"text": gen_prompt}],
        "seed": 0,
        "sampler": "K_EULER_ANCESTRAL",
        "steps": 2
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    ## load back to numpy array 
    print(response_body['artifacts'][0].keys())
    imgdata = base64.b64decode(response_body["artifacts"][0]["base64"])
    filename = os.path.join("C:/Users/zcharpy/Documents/langgraph-example/imgs/",llm_rewrite_to_image_name(prompt))
    print("image is saved to " , filename)
    with open(filename, 'wb') as f:
        f.write(imgdata)   
    im = Image.open(filename)  
    img_location=filename
    return img_location

tools = [generate_image]

