import argparse
import pandas as pd


def read(path):
    df = pd.read_json(path)
    print(f"Read {path} ({len(df)} rows)")
    return df
    

def extract(df):
    def extract(row):
        trips = [(i, label, token)
                 for i, (label, token) in enumerate(zip(row.labels, row.tokens))
                 if label != "O"]        
        return {"document": row.name,
                "token": [x[0] for x in trips],
                "label": [x[1] for x in trips],
                "token_form": [x[2] for x in trips]}

    
    res = pd.json_normalize(df.apply(extract, axis=1)) \
            .explode(["token", "label", "token_form"]) \
            .dropna() \
            .reset_index(drop=True)
    res.index.name = "row_id"
    return res


def save(df, outfile):
    df.to_csv(outfile, header=True, index=True)
    print("Wrote", outfile, f"({len(df)} rows)")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("outfile")
    
    args = p.parse_args()
    df = extract(read(args.infile))
    if args.outfile:
        save(df, args.outfile)
    else:
        print(df.to_csv(header=True, index=True))

    
    
