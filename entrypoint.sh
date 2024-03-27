#!/bin/sh -l
python /main.py \
--openai_api_key "$1" \
--review_type "$2" \
--anthropic_api_key "$3" \
--github_token "$2" \
--github_pr_id "$3" \
--files "$4" \
--openai_model "$5" \
--openai_temperature "$6" \
--openai_max_tokens "$9" \
--logging "$10" \
