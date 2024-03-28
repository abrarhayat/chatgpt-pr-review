#!/bin/sh -l
python /main.py \
--openai_api_key "$1" \
--anthropic_api_key "$2" \
--github_token "$3" \
--github_pr_id "$4" \
--files "$5" \
--openai_model "$6" \
--openai_temperature "$7" \
--openai_max_tokens "$8" \
--review_type "$9"
