from fnmatch import fnmatch
from logging import info, basicConfig, getLevelName, debug
import os
from typing import Iterable, List, Tuple
from argparse import ArgumentParser
from time import sleep
import openai
from openai import OpenAI
from github import Github, PullRequest, Commit
from dotenv import load_dotenv
import requests
from nltk.tokenize import word_tokenize
import tiktoken

load_dotenv()

OPENAI_BACKOFF_SECONDS = 20  # 3 requests per minute
OPENAI_MAX_RETRIES = 3
OLLAMA_API_ENDPOINT = os.getenv("OLLAMA_URL")
openai_client = os.getenv("OPENAI_API_KEY")

AI_SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You're a helpful AI Code Reviewer who is reviewing Capstone Projects for Masters' Students", 
}

messages = [AI_SYSTEM_MESSAGE]

def code_type(filename: str) -> str:
    extension = filename.split(".")[-1].lower()
    if "js" in extension:
        return "JavaScript"
    elif "ts" in extension:
        return "TypeScript"
    elif "java" in extension:
        return "Java"
    elif "py" in extension:
        return "Python"
    else:
        return extension.replace(".", "").upper()


def prompt(filename: str, contents: str) -> str:
    code = "code"
    type = code_type(filename)
    if type:
        code = f"{type} {code}"

    return (
        f"Please evaluate the {code} below inside the triple backticks.\n"
        "Use the following checklist to guide your analysis:\n"
        "   1. Documentation Defects:\n"
        "       a. Naming: Assess the quality of software element names.\n"
        "       b. Comment: Analyze the quality and accuracy of code comments.\n"
        "   2. Visual Representation Defects:\n"
        "       a. Bracket Usage: Identify any issues with incorrect or missing brackets.\n"
        "       b. Indentation: Check for incorrect indentation that affects readability.\n"
        "       c. Long Line: Point out any long code statements that hinder readability.\n"
        "   3. Structure Defects:\n"
        "       a. Dead Code: Find any code statements that serve no meaningful purpose.\n"
        "       b. Duplication: Identify duplicate code statements that can be refactored.\n"
        "   4. New Functionality:\n"
        "       a. Use Standard Method: Determine if a standardized approach should be used for single-purpose code statements.\n"
        "   5. Resource Defects:\n"
        "       a. Variable Initialization: Identify variables that are uninitialized or incorrectly initialized.\n"
        "       b. Memory Management: Evaluate the program's memory usage and management.\n"
        "   6. Check Defects:\n"
        "       a. Check User Input: Analyze the validity of user input and its handling.\n"
        "   7. Interface Defects:\n"
        "       a. Parameter: Detect incorrect or missing parameters when calling functions or libraries.\n"
        "   8. Logic Defects:\n"
        "       a. Compute: Identify incorrect logic during system execution.\n"
        "       b. Performance: Evaluate the efficiency of the algorithm used.\n"
        "Provide your feedback in a numbered list for each category. At the end of your answer, summarize the recommended changes to improve the quality of the code provided.\n"
        f"```\n{contents}\n```"
        "Please provide your feedback within 600 words."
    )


def is_merge_commit(commit: Commit.Commit) -> bool:
    return len(commit.parents) > 1


def files_for_review(
    pull: PullRequest.PullRequest, patterns: List[str]
) -> Iterable[Tuple[str, Commit.Commit]]:
    changes = {}
    commits = pull.get_commits()
    for commit in commits:
        if is_merge_commit(commit):
            info(f"skipping commit {commit.sha} because it's a merge commit")
            continue
        for file in commit.files:
            if file.status in ["unchanged", "removed"]:
                info(
                    f"skipping file {file.filename} in commit {commit.sha} because its status is {file.status}"
                )
                continue
            if not file.patch or file.patch == "":
                info(
                    f"skipping file {file.filename} in commit {commit.sha} because it has no patch"
                )
                continue
            for pattern in patterns:
                if fnmatch(file.filename, pattern):
                    changes[file.filename] = commit
    return changes.items()


def review(
    filename: str, content: str, model: str, temperature: float, max_tokens: int
) -> str:
    if "gpt" in model.lower():
        return review_with_openai(filename, content, model, temperature, max_tokens)
    else:
        return review_with_ollama(filename, content, model, temperature, max_tokens)

def review_with_ollama(filename: str, content: str, model: str, temperature: float, max_tokens: int) -> str:
    try:
        messages.append({"role": "user", "content": prompt(filename, content)})
        # Reset messages if we exceed max tokens
        reset_messages_if_exceeds_max_tokens(filename, content, model, max_tokens)
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(OLLAMA_API_ENDPOINT, json=data, headers=headers, timeout=100)
        chat_review = response.json()['message']['content']
        # print(f"Response for {filename}: \n")
        # print(chat_review)
        # print('\n\n\n')
        messages.append({"role": "assistant", "content": chat_review})
        return f"{model.capitalize()} review for {filename}:*\n" f"{chat_review}"
    except Exception as e:
        print(f'Failed to review file {filename}: {e}')

def review_with_openai(
    filename: str, content: str, model: str, temperature: float, max_tokens: int) -> str:
    x = 0
    global messages, openai_client
    while True:
        try:
            messages.append({"role": "user", "content": prompt(filename, content)})
            # Reset messages if we exceed max tokens
            reset_messages_if_exceeds_max_tokens(filename, content, model, max_tokens)
            chat_review = (
                openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                .choices[0]
                .message.content
            )
            # print(chat_review)
            # print('\n\n\n')
            messages.append({"role": "assistant", "content": chat_review})
            return f"{model.capitalize()} review for {filename}:*\n" f"{chat_review}"
        except openai.RateLimitError:
            if x < OPENAI_MAX_RETRIES:
                info("OpenAI rate limit hit, backing off and trying again...")
                sleep(OPENAI_BACKOFF_SECONDS)
                x+=1
            else:
                raise Exception(
                    f"finally failing request to OpenAI platform for code review, max retries {OPENAI_MAX_RETRIES} exceeded"
                )            
            
def reset_messages_if_exceeds_max_tokens(filename: str, content: str, model: str, max_tokens: int):
    global messages
    concatenated_messages = "".join([message["content"] for message in messages])
    current_token_length = get_token_length_in_words(concatenated_messages, model)
    print(f"{filename}, Token length: " + str(current_token_length))
    if(current_token_length > max_tokens):
        print(f'Resetting messages as tokens are greater than max tokens at {filename}')
        messages = [AI_SYSTEM_MESSAGE]
        messages.append({"role": "user", "content": prompt(filename, content)})


def get_token_length_in_words(text: str, model: str) -> int:
    if("gpt" in model.lower()):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    else:
        tokens = word_tokenize(text)
        return len(tokens)
def main():
    parser = ArgumentParser()
    parser.add_argument("--openai_api_key", default=None, help="OpenAI API Key")
    parser.add_argument("--github_token", default=None, help="Github Access Token")
    parser.add_argument(
        "--github_pr_id", default=None, type=int, help="Github PR ID to review"
    )
    parser.add_argument(
        "--ai_model",
        default=None,
        help="AI model to use. Options: mistral, gpt-3.5-turbo,"
        "text-babbage-001, text-curie-001, text-ada-001. Recommended: gpt-3.5-turbo",
    )
    parser.add_argument(
        "--ai_temperature",
        default=0.5,
        type=float,
        help="Sampling temperature to use, a float [0, 1]. Higher values "
        "mean the model will take more risks. Recommended: 0.5",
    )
    parser.add_argument(
        "--ai_max_tokens",
        default=16384,
        type=int,
        help="The maximum number of tokens to generate in the completion.",
    )
    parser.add_argument(
        "--files",
        default="*",
        help="Comma separated list of UNIX file patterns to target for review",
    )
    parser.add_argument(
        "--logging",
        default="warning",
        type=str,
        help="logging level",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    basicConfig(encoding="utf-8", level=getLevelName(args.logging.upper()))
    global openai_client
    openai_client = OpenAI(api_key=args.openai_api_key if args.openai_api_key else os.getenv("OPENAI_API_KEY"))
    file_patterns = args.files.split(",")
    openai.api_key = args.openai_api_key if args.openai_api_key else os.getenv("OPENAI_API_KEY")
    g = Github(args.github_token if args.github_token else os.getenv("GITHUB_TOKEN"))

    repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
    pull = repo.get_pull(args.github_pr_id if args.github_pr_id else int(os.getenv("GITHUB_PR_ID")))
    comments = []
    files = files_for_review(pull, file_patterns)
    info(f"files for review: {files}")
    for filename, commit in files:
        debug(f"starting review for file {filename} and commit sha {commit.sha}")
        content = repo.get_contents(filename, commit.sha).decoded_content.decode("utf8")
        if len(content) == 0:
            info(
                f"skipping file {filename} in commit {commit.sha} because the file is empty"
            )
            continue
        body = review(
            filename,
            content,
            args.ai_model if args.ai_model else os.getenv("AI_MODEL"),
            args.ai_temperature,
            int(os.getenv("AI_MAX_TOKENS")) if os.getenv("AI_MAX_TOKENS") else args.ai_max_tokens,
        )
        if body != "":
            debug(f"attaching comment body to review:\n{body}")
            comments.append(
                {
                    "path": filename,
                    # "line": line,
                    "position": 1,
                    "body": body,
                }
            )

    if len(comments) > 0:
        model = args.ai_model if args.ai_model else os.getenv("AI_MODEL")
        pull.create_review(
            body=f"**{model} code review**", event="COMMENT", comments=comments
        )


if __name__ == "__main__":
    main()
