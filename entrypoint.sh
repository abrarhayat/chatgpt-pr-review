#!/bin/sh -l
python /main.py \
--openai_api_key "$1" \
--review_type "$2" \
--anthropic_api_key "$3" \
--github_token "$4" \
--github_pr_id "$5" \
--files "$6" \
--openai_model "$7" \
--openai_temperature "$8" \
--openai_max_tokens "$9" \
--logging "$10" \
