import argparse
import os
from pathlib import Path
from pprint import pp
import random
import sys
import textwrap
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import google.generativeai as genai

print(f"Running python {sys.version}")
print(f"generativeai: {genai.__version__}")

RNG_SEED = random.randint(0, 2718281828)        

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-pro"


def init_model(temperature=0.8, top_p=0.95, top_k=60):
    # See: https://ai.google.dev/docs/gemini_api_overview
    # https://ai.google.dev/api/python/google/generativeai/GenerationConfig

    config = {
        "candidate_count": 1,  # default (>1 may raise an exception)
        "stop_sequences": None,
        "max_output_tokens": 4096,  # seems largely to be ignored!
        "temperature": temperature,  # 0 <= t <= 1.0, closer to 1.0 is more random/"creative" output
        "top_p": top_p,  # maximum cumulative probability of tokens to consider when sampling
        "top_k": top_k,  # defaults to 40 (maximum number of tokens to be considered when sampling)
    }

    generation_config = genai.types.GenerationConfig(**config)
    model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)

    print(f"Using model {model.model_name!r}")
    print(f"Generation config:")
    pp(config)
    print()

    return model


#
# first names - year-of-birth 2000 - sorted by frequency
#
FIRST_NAMES = pd.read_csv(
    "../datasets/yob2000.txt", keep_default_na=False, names=("name", "gender", "freq")
)
FIRST_NAMES = FIRST_NAMES[
    FIRST_NAMES.freq >= 100
]  # 3056 most frequent ones; 1299 male, 1757 female

n = sum(FIRST_NAMES.freq)
FIRST_NAMES["p"] = FIRST_NAMES.freq / n

#
# last names - drop last row ("ALL OTHER NAMES") - others are sorted by frequence
#
SURNAMES = pd.read_csv(
    "../datasets/Names_2010Census.csv", header=0, keep_default_na=False
)[:-1]
SURNAMES.name = SURNAMES.name.apply(str.title)
SURNAMES = SURNAMES.iloc[:10_000]

n = sum(SURNAMES["count"])
SURNAMES["p"] = SURNAMES["count"] / n


TOPICS = (
    "visualization",
    "brainstorming",
    "storytelling",
    "mind mapping",
    "learning launch",
)


def fake(n=10):
    assert n < 1100  # limited by size of first_names
    n = max(n, 100) 
    
    first_names = np.random.choice(FIRST_NAMES.name, size=3 * n, p=FIRST_NAMES.p)
    last_names = np.random.choice(SURNAMES.name, size=3 * n, p=SURNAMES.p)
    
    full_names = [f"{a} {b}" for (a, b) in zip(first_names, last_names)]
    full_names1 = full_names[n:] + full_names[:n]
    full_names2 = full_names[2 * n :] + full_names[: 2 * n]

    # make sure that all prof names are shifted at least by 3 from full_names
    # in the same doc we never want a student name to be equal to a prof name
    
    prof = (
        "teacher, professor Jeanne Liedtka",
        "teacher, professor Liedtka",
        "teacher, Dr. Liedtka",
    ) * n
    prof1 = full_names[-10:] + full_names[:-10]
    prof2 = full_names[-20:] + full_names[:-20]

    book = ("the Genji Monogatari", "the Tao Te Ching", "the Bible") * n
    college = ("Garvard", "Harvard", "Stanford") * n
    
    df = pd.DataFrame(
        dict(
            first_name=first_names,
            last_name=last_names,
            student=full_names,
            fellow1=full_names1,
            fellow2=full_names2,
            prof=prof,
            prof1=prof1,
            prof2=prof2,
            book=book,
            college=college
        )
    )

    return df[:n]


def generate(template, model, n=10):
    template_name, template = template

    if template_name == "template6":
        # set random seed back to original one!
        #
        # this will force some names that were earlier used as student names
        # now to be used as non-student names in a context that makes clear
        # they are a not students
        #
        random.seed(RNG_SEED)
        np.random.seed(RNG_SEED)
    
    fake_data = fake(n)

    res = []

    for i in tqdm(range(n)):
        topic = TOPICS[i % len(TOPICS)]
        data = fake_data.iloc[i]

        prompt = template.format(
            topic=topic,
            title=topic.title(),
            name=data.student,
            name1=data.fellow1,
            name2=data.fellow2,
            prof=data.prof,
            prof1=data.prof1,
            prof2=data.prof2,
            book=data.book,
            college=data.college,
        )

        try:
            start = time.time()
            resp = model.generate_content(prompt)
            end = time.time()
        except Exception as exc:
            sys.stderr.write(f"[{i}] generate_content: ignoring {exc}\n")
            time.sleep(10)
            continue

        parts = resp.parts
        if len(parts) > 0:
            text = resp.parts[0].text
        else:
            try:
                text = resp.text
            except Exception as exc:
                sys.stderr.write(f"[{i}] extracting text: ignoring {exc}\n")
                continue

        res.append((template_name, topic, text, end - start))
        time.sleep(1.0 / 60)  # prevent rate limiting

    return res


def read_templates(template_dir):
    for path in sorted(Path(template_dir).glob("template*.txt")):
        with open(path) as f:
            yield (path.stem, f.read())


def save(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, header=True, index=False, sep="\t")
    print(f"Wrote {path} ({len(df)} rows)")


def run(model, indir, outdir, n=10):
    start = time.time()
    dfs = []
    for template_name, template in read_templates(indir):
        res = generate((template_name, template), model=model, n=n)
        df = pd.DataFrame(res, columns=("template", "topic", "text", "time"))
        save(df, Path(outdir) / f"{MODEL_NAME}-{template_name}.tsv")

    end = time.time()
    print(f"Total time: {end - start:.1f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--indir",
        help="Input dir (templates) (defaults to 'prompts')",
        default="prompts",
    )
    p.add_argument(
        "-o", "--outdir", help="Output dir (defaults to 'output')", default="output"
    )
    p.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate (defaults to 10)",
    )
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument(
        "-t",
        "--temp",
        type=float,
        default=0.8,
        help="Model temperature (0 .. 1.) (defaults to 0.8)",
    )

    args = p.parse_args()

    RNG_SEED = args.seed or random.randint(0, 2718281828)        
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    model = init_model(temperature=args.temp)
    run(model, args.indir, args.outdir, n=args.num_samples)
