from transformers import AutoTokenizer, GPT2TokenizerFast


def main():
    REPOS = [
        # "facebook/opt-125m",
        # "facebook/opt-350m",
        # "facebook/opt-1.3b",
        # "facebook/opt-2.7b",
        # "facebook/opt-6.7b",
        # "facebook/opt-13b",
        # "facebook/opt-30b",
        # "facebook/opt-66b",
        # "facebook/opt-iml-1.3b",
        # "facebook/opt-iml-max-1.3b",
        # "facebook/opt-iml-30b",
        # "facebook/opt-iml-max-30b",

        "huggingface/opt-175b",
        "huggingface/opt-iml-175b",
        "huggingface/opt-iml-max-175b"
    ]

    for repo_name in REPOS:
        tokenizer = GPT2TokenizerFast.from_pretrained(repo_name)
        assert tokenizer.is_fast
        tokenizer.push_to_hub(
            # "TimeRobber/opt-tokenizer",
            repo_name,
            commit_message="Add fast tokenizer",
            create_pr=True,
            use_auth_token=True
        )
        # tokenizer.save_pretrained(
        #     repo_name,
        #     push_to_hub=True
        # )

if __name__ == "__main__":
    main()