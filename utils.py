from openai import OpenAI
from IPython.display import display, Markdown
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

api_key = os.getenv('OPENAI_API_KEY')

def display_responses(*args):
    markdown_string = "<table><tr>"
    # Headers
    for arg in args:
        markdown_string += f"<th>System Prompt:<br />{arg['system_prompt']}<br /><br />"
        markdown_string += f"User Prompt:<br />{arg['user_prompt']}</th>"
    markdown_string += "</tr>"
    # Rows
    markdown_string += "<tr>"
    for arg in args:
        markdown_string += f"<td>Response:<br />{arg['response']}</td>"
    markdown_string += "</tr></table>"
    display(Markdown(markdown_string))



MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_completion(system_prompt, user_prompt, model=MODEL):
    """"
    Get a completion from the OpenAI API using the specified system and user prompts.
    
    Args:
        system_prompt (str): The system prompt to set the context for the model.
        user_prompt (str): The user prompt containing the main query or instruction.
        model (str): The model to use for generating the completion. Default is "gpt-4o-mini".
    Returns:
        The completion text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None   