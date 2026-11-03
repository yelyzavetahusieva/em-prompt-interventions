"""The actual call to the judge model (Claude Haiku 5.5).

Kept separate from judge.py so that judge.py never depends on a provider.
To use a different judge model, write another function with the same shape:
it takes a prompt string and returns the judge's reply as a string.

Needs an API key: put ANTHROPIC_API_KEY=... in the .env file in the repo root
(it is gitignored, so it never gets pushed to GitHub).
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()  # copies the lines of .env into the environment (does nothing if no .env)

# The one fixed judge for the whole experiment. Saved alongside every score.
JUDGE_MODEL = "claude-haiku-5-5"

# Created once and reused. It reads ANTHROPIC_API_KEY from the environment and
# automatically retries rate-limit and server errors.
_client = anthropic.Anthropic()


def ask_haiku(prompt):
    """Send one prompt to the judge and return its text reply.

    Returns "" if the model declined to answer. An empty reply fails parsing in
    judge.py, so it is counted as a parse failure instead of crashing the run.
    """
    response = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2048,  # room for the short JSON answer plus a little thinking
        output_config={"effort": "low"},  # keep thinking small; judging is a simple task
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return ""
    # The reply is a list of blocks (thinking, text, ...). We only want the text.
    return "".join(block.text for block in response.content if block.type == "text")
