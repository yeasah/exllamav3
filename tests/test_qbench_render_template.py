import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Tokenizer
from exllamav3.util.misc import prepend_hf_chat_context

# qbench `template: render`: a test row must sit exactly where the chat template puts an
# assistant reply's content. The reference is the template's own tokenized render of a finished
# reply -- the prefix has to be a literal prefix of it, BOS included or not as the template
# decides. Each tokenizer is one way the cheaper modes got this wrong:
#   Qwen3.6 / Ornith: "generation" leaves the row inside an opened <think> block, and
#       "assistant" (continue_final_message) drops the "\n\n" after </think>
#   gpt-oss: "generation" puts the row where <|channel|> belongs
#   gemma-4: BOS comes from the template, not the tokenizer
#   Qwen3-0.6B: "generation" omits the empty think block the template renders
REPOS = [
    "Qwen/Qwen3.6-35B-A3B",
    "ornith-ai/Ornith-1.5-35B-A3B",
    "openai/gpt-oss-20b",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3-0.6B",
]
BODY = "A magazine supplement with an image of a lighthouse and the title"
MESSAGES = [{"role": "system", "content": ""}, {"role": "user", "content": "Say something."}]


def _tokenizer(repo):
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(repo, local_files_only = True,
                                 allow_patterns = ["*.json", "*.jinja", "*.model", "*.txt"])
        return Tokenizer.from_config(Config.from_directory(path))
    except Exception:
        return None


class RenderTemplateTest(unittest.TestCase):

    def test_prefix_is_the_templates_own_reply_prefix(self):
        tested = 0
        for repo in REPOS:
            tok = _tokenizer(repo)
            if tok is None:
                continue
            tested += 1
            with self.subTest(repo = repo):
                body = tok.encode(BODY)
                ids = prepend_hf_chat_context(tok, body, mode = "render")[0].tolist()
                prefix = ids[:len(ids) - body.shape[-1]]
                full = tok.hf_chat_template(
                    MESSAGES + [{"role": "assistant", "content": BODY}],
                    add_generation_prompt = False,
                )[0].tolist()
                self.assertEqual(full[:len(prefix)], prefix)
                # and the reply content follows the prefix directly in the real render
                rest = tok.decode(torch.tensor(full[len(prefix):]), decode_special_tokens = True)
                self.assertTrue(rest.startswith(BODY), repr(rest[:80]))
        if not tested:
            self.skipTest("no reference tokenizer in the local HF cache")


if __name__ == "__main__":
    unittest.main()
