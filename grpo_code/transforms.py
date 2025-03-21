SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

Additionally, you may optionally use the following imports:

import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from typing import List, Dict, Tuple, Optional, Any

If you choose to use any of these imports, ensure they are included
inside the <answer> </answer> tags, e.g:

<answer>
import time
import math
...
</answer>

You may not utilise any other imports or filesystem operations. 
"""

def axolotl_acecode_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [{"role": "user", "content": example["question"] + "\n\n" + SYSTEM_PROMPT}],
            "answers": example["test_cases"],
        }
    return transform_fn, {"remove_columns": ["question", "test_cases"]}
