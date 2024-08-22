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

from langgraph.graph import END, StateGraph, MessageGraph
from typing import List, Tuple, Annotated, TypedDict , Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field, validator


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
    filename = 'output.jpg'
    print("image is saved to " , filename)
    with open(filename, 'wb') as f:
        f.write(imgdata)   
    im = Image.open(filename)  
    img_location=filename
    return img_location



## structural output using LMFE 
class ScenePlot(BaseModel):     
    story_plot: str = Field(description="story plot for the scene")
    characters : List[str] = Field(description="a list of the name of characters in this story plot")
    background_story : List[str] = Field(description="a list of the background story for each character, ordered and aligned with the name listed above.")
    theme: str = Field(description="the central theme of the story")
    genre: str = Field(description="genre of the scene usually sci-fi, drama, horrow, comedy, thriller, romance and so on")

llm_with_sceneplot=llm.with_structured_output(ScenePlot)     

scene_creation_prompt = ChatPromptTemplate.from_template("""Act as an experienced screen scriptwriter.
    Develop one linear plotline, character arcs, or unique concepts based on the given [theme] and [genre]. 
    You are to develop one scene only. Explore innovative angles and replace human characters with animal characters.
    Ensure you create up to maximum 3 characters for the entire story.
    The entire story should be linear, short, compact and compelling for a wide audience. 
    Divulge just enough info to provide a clear direction of the story line.

    Ensure to include brief plot, character arcs and
    
    You will need to specify the below into your scene scripts :
    -----
    story_plot : what this story is about
    characters : a list of the name for characters ordered by their appearance in the storyline.
    background_story : a list background story for each character, ordered and aligned with the name listed above.
    theme : what is the theme
    genre : what is the genre
    
    -----
    movie director input :
    theme: {theme} 
    genre: {genre} """)
                                              
scene_chain = scene_creation_prompt | llm_with_sceneplot

def story_creator(prompt):
    none_output=True
    while none_output ==True :
        output=scene_chain.invoke({"theme": "write me a short story about the friendship between a dragon and a little girl" , "genre":"comedy"})
        if type(output) is not None:
            none_output=False
            break
    char_bk='\n'.join([(c,b) for (c,b) in zip(output.characters, output.background_story)])
    story_output=f"The theme of the story is :{output.theme} \n the plot of the story is : { output.story_plot} \n . There are the characters and their background stories : {char_bk}"
    return story_output
tools = [generate_image]

