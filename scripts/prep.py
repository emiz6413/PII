# !pip install spacy
# !pip install spacy_alignments
# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_lg

import argparse
from pathlib import Path
import pandas as pd
import regex as re
import spacy
import spacy_alignments as alignments
from tqdm import tqdm

import extract

MODEL_NAME = "en_core_web_lg"

markup1 = re.compile(r"^\*\*([^\*]+)\*\*$", re.MULTILINE)
markup2 = re.compile(r"\*\*([^\*]+)\*\*:")
tag = re.compile(r"</?\w+>")
student_name = re.compile(r"<NAME>([^<>]+)</NAME>")
nested_xml = re.compile(r"(<URL>[^<>]+)<USERNAME>([^<>]+)</USERNAME>([^<>]*</URL>)")
empty_xml = re.compile(r"<(?P<tag>\w+)>(?P<content>[^<>]{0,2})</\1>")
xml = re.compile(r"<(?P<tag>\w+)>(?P<content>[^<>]{2,200})</\1>")


model = spacy.load(MODEL_NAME)


def main(indir, outdir):
    indir = Path(indir)
    outdir = Path(outdir)

    assert indir.is_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path in sorted(indir.glob("*.tsv")):
        dfs.append(prep(path))

    df = pd.concat(dfs).dropna()
    pii = extract.extract(df)
    
    df.to_json(outdir / "train.json", orient="records")
    pii.to_csv(outdir / "sample_submission_raw.csv", header=True, index=True)
    print(f"Wrote {outdir}/train.json ({len(df)} rows)")
    print(f"Wrote {outdir}/sample_submission_raw.csv ({len(pii)} rows)")
            

def prep(path):
    print(f"Prepping {path}...")
    df = pd.read_csv(path, header=0, sep="\t")
    prepped = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prepped.append(prep_text(row))
    # prepped = df.apply(prep_text, axis=1)
    return pd.json_normalize(prepped)


def prep_text(row):

    # construct document id
    if "template" not in row.index:
        n = 0
    else:
        try:
            n = int(re.search(r"\d+", row.template).group())
        except:
            n = 0
    document = 1000 * n + row.name
    
    text, clean_text, matches = normalize(row.name, row.text)
    
    if not clean_text:
        return {"document": document,
                "full_text": None,
                "tokens": None,
                "trailing_whitespace": None,
                "labels": None}
        
    doc = model(clean_text)
    tokens = [token.text for token in doc]
    trailing_whitespace = [bool(token.whitespace_) for token in doc]
    try:
        labels =  align(document, tokens, trailing_whitespace, matches, text, clean_text)
    except ValueError:
        print(f"Alignment failed for row {row.name}/document {document}")
        labels = None

    # extra info could also be usefull for training
    # can also be constructed in testing
    # pos_ = [token.pos_ for token in doc]  # PUNCT, ...
    # pos = [token.pos for token in doc] # numerical
    # is_oov = [int(token.is_oov) for token in doc]    
    # is_digit
    # like_email
    # like_url
    # like_num

    return {"document": document,
            "full_text": clean_text,
            "tokens": tokens,
            "trailing_whitespace": trailing_whitespace,
            "labels": labels}


def normalize(index, text):
    # remove markdown
    text = markup1.sub(r"\1", text)
    text = markup2.sub(r"\1", text)

    # try to work around garbage in the gemini output
    # template3 - no problems; template4 and 6 just a few;
    # template5.txt causes most problems
    
    # (1) gemini has a tendency to generate nested XML in URLs :/
    # this removes at least the extra <USERNAME> tag
    text = nested_xml.sub(r"\1\2\3", text)

    # (2) deal with this <URL>https://harvard.edu/~<USERNAME>/</URL>
    m = student_name.search(text)
    if m:
        username = "~" + m.group(1).lower().replace(" ", "_")        
        text = text.replace("~<USERNAME></USERNAME>", username)
        text = text.replace("~<USERNAME>", username)
        text = text.replace("/<USERNAME>", username)
        
    
    # (3) sometimes gemini also adds incorrect XML tags
    # or other garbage like "<NAME></NAME><NAME></NAME>"
    text = empty_xml.sub("", text)

    # prevent very weird tokenizations by Spacy!
    text = text.replace("](", "] (").replace("]<", "] <")
                          
    # remove the class tags
    matches = list(xml.scanner(text))
    clean_text = xml.sub(r"\2", text)

    # if there are any remaining tags, gemini messed up
    # return empty string, so we'll skip this doc :/
    m = tag.search(clean_text)
    if m:
        print(f"Warning: Skipping row {index} because of isolated XML tag: {m}")
        return text, "", None
    
    return text, clean_text, matches


def align(document, tokens, trailing_whitespace, matches, text, clean_text):
    def reconstruct_text(start, end):
        return "".join(token + " " * ws
                       for (token, ws) in zip(tokens[start:end], trailing_whitespace[start:end]))
    
    labels = ["O" for _ in tokens]
    
    if not matches:
        return labels

    # alignment needs to be done in two steps, since get_alignments is not totally reliable!
    # also, the spacy tokenizer makes weird tokenizations which can lead to problems
    # (e.g. "....URL](https://www.xyz.edu/..." is tokenized as ['URL](htts://www.xyz.edu', '/'])
    
    # char -> char
    c2c, _ = alignments.get_alignments(list(text), list(clean_text))

    # char -> token
    c2t, _ = alignments.get_alignments(list(clean_text), tokens)    
    
    for m in matches:
        tag, content = m.groups()
        b_label, i_label = get_labels(tag)

        start, end = convert_span(m.span(), c2c)
        start, end = convert_span((start, end), c2t)
        
        assert 0 <= start < end <= len(tokens)

        labels[start] = b_label
        for i in range(start + 1, end):                
            labels[i] = i_label

        # double-check
        txt = reconstruct_text(start, end)
        
        if txt.strip() != content.strip():
            start1, end1 = m.span()
            start1 = max(start1 - 100, 0)
            end1 = end1 + 80
            print(f"[Document {document}] ERROR for {m}")
            print(f"Raw context: {text[start1:end1]!r}")
            print(f"Text:  {txt.encode('utf-8')}")  # encoding as binary to show some errors better
            print(f"Match: {content.encode('utf-8')}")
            print(f"Tokens: {tokens[start:end]}")
            print(f"Labels: {labels[start:end]}")
            raise ValueError("Alignment")
            
    return labels


def convert_span(span: tuple[int, int], alignment: list[list[int]]) -> tuple[int, int]:
    # alignment should be an alignment returned by alignments.get_alignment
    start, end = span
    if start < 0 or end < 0:
        return -1, -1
    indices = [i for ilis in alignment[start:end] for i in ilis]
    if not indices:
        return -1, -1
    return (indices[0], indices[-1] + 1)
    
    

def get_labels(tag):
    mm = {
        "NAME": ("B-NAME_STUDENT",
                 "I-NAME_STUDENT"),
        "NAME_STUDENT": ("B-NAME_STUDENT",
                         "I-NAME_STUDENT"),
        "EMAIL": ("B-EMAIL",   
                  "B-EMAIL"), # singles only
        "URL": ("B-URL_PERSONAL",
                "I-URL_PERSONAL"),
        "ADDRESS": ("B-STREET_ADDRESS",
                    "I-STREET_ADDRESS"),
        "PHONE_NUM": ("B-PHONE_NUM",
                      "I-PHONE_NUM"),
        "ID_NUM": ("B-ID_NUM",
                   "I-ID_NUM"),
        "USERNAME": ("B-USERNAME",
                     "B-USERNAME"), # singles only
    }
    assert tag in mm, tag
    return mm.get(tag, (tag, tag))
        
    
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--indir", default="output")
    p.add_argument("-o", "--outdir", default="output")

    args = p.parse_args()
    main(args.indir, args.outdir)
    
    
    
