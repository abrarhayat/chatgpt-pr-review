from fnmatch import fnmatch
from logging import info, basicConfig, getLevelName, debug
from time import sleep
import os
from typing import Iterable, List, Tuple, Optional
from argparse import ArgumentParser
from re import search
import openai
from github import Github, PullRequest, Commit

OPENAI_BACKOFF_SECONDS = 20  # 3 requests per minute
OPENAI_MAX_RETRIES = 3


def code_type(filename: str) -> str | None:
    match = search(r"^.*\.([^.]*)$", filename)
    if match:
        match match.group(1):
            case "js":
                return "JavaScript"
            case "ts":
                return "TypeScript"
            case "java":
                return "Java"
            case "py":
                return "Python"


def prompt(filename: str, contents: str, review_type: str) -> str:
    if review_type == '':
        review_type = "checklist"
    code = "code"
    type = code_type(filename)
    if type:
        code = f"{type} {code}"
    if(review_type == "checklist"):
        return (
            f"Please evaluate the {code} below.\n"
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
        )
    elif(review_type == 'scalability'):
        return (
            f"Please evaluate the {code} below.\n"
            "Use the following checklist to guide your analysis:\n"
            "Configuration: For the following configuration of the server, Intel Xeon 2.4 GHZ, 2 GB Memory, 4 x 60 GB HDD\n"
            "For the given configuration, how does the code change contribute to scalability in the following criteria:\n"
            "   1. Efficiency of the algorithm\n"
            "       a. Time Complexity\n"
            "       b. Space Complexity\n"
            "       c. Data Structure used\n"
            "   2. Maintainability of code for further improvements\n"
            "       a. Readability of code\n"
            "       b. Reusability of code\n"
            "       c. How well can the functionalities be extended\n"
            "   3. User Requests\n"
            "       a. Number of requests that can be served without making too many changes to the code\n"
            "       b. Speed of response to incoming requests given memory constraints\n"
            "       c. How well will it scale in terms of disk space\n"
            "   4. Debugging/Testing\n"
            "       a. How well will can it be debugged or tested\n"
            "       b. How adjustable is the code to creating new integration and regression tests\n"
            "       c. How much more complexity does it add to performing UI automation tests\n"
            "       d. Generate unit tests at the end of this review\n"
            "   5. Error Handling\n"
            "       a. How well can it handle exceptions\n"
            "       b. Are proper handling methods used\n"
            "       c. Can some errors be handled before they occur\n"
            "Provide your feedback in a numbered list for each category. At the end of your answer, summarize the recommended changes to improve the quality of the code provided.\n"
            f"```\n{contents}\n```"
        )
    elif(review_type == 'performance'):
        return (
            f"Please evaluate the {code} below.\n"
            "Use the following checklist to guide your analysis:\n"
            "Has the improved changes contribute to performance?\n"
            "if so comment on the performance\n"
            f"```\n{contents}\n```"
        )
    elif(review_type == 'uml'):
        return (
            f"Please evaluate the {code} below.\n"
                "Use the following checklist to guide your analysis:\n"
                "Please comment on the structural changes made\n"
                "If everything is fine, generate PlantUML for the changes, otherwise correct the changes and then generate PlantUML\n"
                "If the file itself was PlantUML, improve the PlantUML\n"
            f"```\n{contents}\n```"
        )
    elif(review_type == 'tdd'):
        return (
            f"Please evaluate the {code} below and generate tests on based tdd practices based on the given code. Please only generate code and annotate the tests:\n"
            f"Read through the file and determine it is already a test file, then make an attempt to improve it, otherwise generate unit tests for the file keeping in mind that."
            f"Make sure to suggest the best practices of keeping the tests in a separate file so that all the test files can be run together as part of CI/CD when the code is pushed.\n"
            f"```\n{contents}\n```"
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
    filename: str, content: str, model: str, temperature: float, max_tokens: int, review_type: str
) -> str:
    x = 0
    while True:
        try:
            chat_review = (
                openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt(filename, content, review_type=review_type),
                        }
                    ],
                )
                .choices[0]
                .message.content
            )
            return f"*ChatGPT review for {filename}:*\n" f"{chat_review}"
        except openai.error.RateLimitError:
            if x < OPENAI_MAX_RETRIES:
                info("OpenAI rate limit hit, backing off and trying again...")
                sleep(OPENAI_BACKOFF_SECONDS)
                x+=1
            else:
                raise Exception(
                    f"finally failing request to OpenAI platform for code review, max retries {OPENAI_MAX_RETRIES} exceeded"
                )


def main():
    parser = ArgumentParser()
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--github_token", required=True, help="Github Access Token")
    parser.add_argument(
        "--github_pr_id", required=True, type=int, help="Github PR ID to review"
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-3.5-turbo",
        help="GPT-3 model to use. Options: gpt-3.5-turbo, text-davinci-002, "
        "text-babbage-001, text-curie-001, text-ada-001. Recommended: gpt-3.5-turbo",
    )
    parser.add_argument(
        "--openai_temperature",
        default=0.5,
        type=float,
        help="Sampling temperature to use, a float [0, 1]. Higher values "
        "mean the model will take more risks. Recommended: 0.5",
    )
    parser.add_argument(
        "--openai_max_tokens",
        default=2048,
        type=int,
        help="The maximum number of tokens to generate in the completion.",
    )
    parser.add_argument(
        "--files",
        help="Comma separated list of UNIX file patterns to target for review",
    )
    parser.add_argument(
        "--logging",
        default="warning",
        type=str,
        help="logging level",
        choices=["debug", "info", "warning", "error"],
    )

    parser.add_argument(
        "--review_type",
        default='',
        type=str,
        help="review type",
        choices=["uml", "scalability", "performance", 'tdd', ''], # Leave empty for default checklist based review
    )
    args = parser.parse_args()

    basicConfig(encoding="utf-8", level=getLevelName(args.logging.upper()))
    file_patterns = args.files.split(",")
    openai.api_key = args.openai_api_key
    g = Github(args.github_token)

    repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
    pull = repo.get_pull(args.github_pr_id)
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
        if(filename.endswith(".config")):
            info(
                f"skipping file {filename} in commit {commit.sha} because the file is a config file"
            )
            continue
        if(filename.endswith(".puml")):
            body = review(
            filename,
            content,
            args.openai_model,
            args.openai_temperature,
            args.openai_max_tokens,
            'uml')
        else:   
            body = review(
                filename,
                content,
                args.openai_model,
                args.openai_temperature,
                args.openai_max_tokens,
                args.review_type
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
        pull.create_review(
            body="**ChatGPT code review**", event="COMMENT", comments=comments
        )


if __name__ == "__main__":
    main()
