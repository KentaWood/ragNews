#!/bin/python3

from groq import Groq
import os

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def extract_keywords(text, seed=None):
    r'''
    This is a helper function for RAG.
    Given an input text,
    this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Who is the current democratic presidential nominee?', seed=0)
    'Joe Biden Presidential Democratic Nominee'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'Trump Mexico border immigrants illegal immigration enforcement deportation'

    Note that the examples above are passing in a seed value for deterministic results.
    In production, you probably do not want to specify the seed.
    '''

    # FIXME:
    # Implement this function.
    # It's okay if you don't get the exact same keywords as me.
    # You probably certainly won't because you probably won't come up with the exact same prompt as me.
    # To make the test cases above pass,
    # you'll have to modify them to be what the output of your prompt provides.

    system = "What are ten words about this topics or question, give only the keywords of this, get rid of \",\" and \"Here are the ten keywords:\" and write them all in one line with a space inbetween each word in the output"
    return run_llm(system,text,seed=seed)

def run_llm(system, user, model='llama3-8b-8192', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                'role': 'user',
                'content': user,
            },
        ],
        model=model,
        seed=seed,
    )
    return chat_completion.choices[0].message.content


if __name__ == '__main__':
    pass