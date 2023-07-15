from pathlib import Path
import pandas as pd

# Here we will read the train data from csv file and store it in a dataframe
if __name__ == "__main__":
    data_dir = Path("./data")
    prompts_train_df = pd.read_csv(data_dir / "prompts_train.csv")
    summaries_train_df = pd.read_csv(data_dir / "summaries_train.csv")
    df = summaries_train_df.merge(prompts_train_df)
    df["excerpt"] = (
        "<PROMPT_TITLE>"
        + df.prompt_title
        + "</PROMPT_TITLE>\n\n"
        + "<PROMPT>"
        + df.prompt_question
        + "</PROMPT>\n\n"
        + "<SUMMARY>"
        + df.text
        + "</SUMMARY>"
    )
    df.to_csv("./data/train.csv", index=False)
