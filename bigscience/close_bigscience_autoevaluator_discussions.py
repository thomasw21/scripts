from huggingface_hub import change_discussion_status, get_repo_discussions

def main():
    repo_ids = [
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
        "bigscience/bloom",
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1",
        "bigscience/bloomz-1b7",
        "bigscience/bloomz-3b",
        "bigscience/bloomz-7b1",
        "bigscience/bloomz",
    ]
    autoevaluator_name = "autoevaluator"
    for repo_id in repo_ids:
        discussions = get_repo_discussions(
            repo_id=repo_id
        )
        for discussion in discussions:
            if discussion.status == "closed":
                # Already closed, don't care
                continue

            if discussion.author != autoevaluator_name:
                continue

            print(discussion.title)
            change_discussion_status(
                repo_id=repo_id,
                discussion_num=discussion.num,
                new_status="closed",
                comment="Closing autoevaluation PRs. Please re-open and ping me if you think this is a mistake."
            )

if __name__ == "__main__":
    main()
