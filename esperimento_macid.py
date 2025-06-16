import sqlite3
import json
import argparse
from itertools import product
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
from unidecode import unidecode

def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

coppie_esempi_json = load_jsonl('../nuovi_jsonl/macid_couples_examples.jsonl')
quadruple_esempi_json = load_jsonl('../nuovi_jsonl/macid_dataset_huggingface_examples.jsonl')

def dummy_processor(prompt, model, key, endpoint):
    return '1'

def oai_processor(prompt, model, key='EMPTY', endpoint='http://localhost:8000/v1', openrouter_noquant=True, openrouter_noreasoning=True):
    client = OpenAI(api_key=key, base_url=endpoint)
    extra_body=None
    if openrouter_noquant:
        if extra_body is None:
            extra_body={}
        extra_body["provider"]={"require_parameters": True, "quantizations":['bf16','fp16','fp32']}
    if openrouter_noreasoning:
        if extra_body is None:
            extra_body={}
        extra_body["reasoning"]={'exclude':True, 'enabled':False}
    try:
        completion = client.completions.create(model=model, prompt=prompt, extra_body=extra_body, temperature=0, seed=27, max_tokens=4)
        return completion.choices[0].text
    except Exception as e:
        print(f"{e}")
        return f"{str(e)}"

def genera_prompt(doc, task, tipo_prompt=None, fewshot=False):
    with open("istruzioni.json") as f:
        istruzioni = json.load(f)
    prompt = istruzioni[task][tipo_prompt or 0] + "\n\n"

    if fewshot:
        esempi = coppie_esempi_json if task == "coppie" else quadruple_esempi_json
        for es in esempi:
            if task == "coppie":
                prompt += f"1) {es['entry_1_sent']}\n2) {es['entry_2_sent']}\nRisposta: {'Sì' if es['different']==0 else 'No'}\n"
            else:
                prompt += f"1) {es['s1']}\n2) {es['s2']}\n3) {es['s3']}\n4) {es['s4']}\nIntruso: {es['intruder']}\n"

    if task == "coppie":
        prompt += f"1) {doc['entry_1_sent']}\n2) {doc['entry_2_sent']}\nRisposta: "
    else:
        prompt += f"1) {doc['s1']}\n2) {doc['s2']}\n3) {doc['s3']}\n4) {doc['s4']}\nIntruso: "
    return prompt

def macid_prompt_variants():
    return [
        (f"macid_{p}_{task}_{'fewshot' if fs else 'zeroshot'}", p, task, fs)
        for p, task, fs in product(range(3), ['coppie', 'quadruple'], [True, False])
    ] + [
        ('macid_onlyfewshot_coppie', None, 'coppie', True),
        ('macid_onlyfewshot_quadruple', None, 'quadruple', True)
    ]

def eval_macid(model, key, endpoint, db_filename, openrouter_noquant, openrouter_noreasoning):
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM coppie")
    num_coppie = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM quadruple")
    num_quadruple = cur.fetchone()[0]
    conn.close()

    variants = macid_prompt_variants()
    total_evals = sum(num_coppie if task == 'coppie' else num_quadruple for _, _, task, _ in variants)

    with tqdm(total=total_evals, desc="Overall Experiment Progress") as pbar:
        for label, tipo_prompt, task, fewshot in variants:
            table = 'coppie' if task == 'coppie' else 'quadruple'
            eval_table = f"{table}_evaluations"
            keys = (
                ['id', 'row_id', 'entry_1_id', 'entry_1_ac', 'entry_1_sent', 'entry_1_video',
                 'entry_2_id', 'entry_2_ac', 'entry_2_sent', 'entry_2_video', 'different']
                if task == 'coppie'
                else ['id', 's1', 'v1', 's2', 'v2', 's3', 'v3', 's4', 'v4', 'intruder']
            )

            conn = sqlite3.connect(db_filename)
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            processor=oai_processor
            for row in rows:
                doc = dict(zip(keys, row))
                prompt = genera_prompt(doc, task, tipo_prompt=tipo_prompt, fewshot=fewshot)
                resp = processor(prompt, model=model, key=key, endpoint=endpoint,openrouter_noquant=openrouter_noquant, openrouter_noreasoning=openrouter_noreasoning)
                acc,expected = evaluate_response(resp, doc, task)
                now = datetime.now()

                id_col_name = 'coppia_id' if task == 'coppie' else 'quadrupla_id'
                cur.execute(f"""
                    INSERT INTO {eval_table} ({id_col_name}, model, prompt, answer,expected, accuracy, prompt_type, fewshot, response_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc['id'], model, prompt, resp.strip(),expected, acc, tipo_prompt, fewshot, now))
                pbar.update(1)
            conn.commit()
            conn.close()

def evaluate_response(resp, doc, task):
    if task== "coppie":
        expected = 'si' if doc['different'] == 0 else 'no'
    else:
        expected=str(doc['intruder'])
    cleaned = unidecode(resp.strip().lower())
    if not cleaned: return 0,expected
    return 1 if cleaned.startswith(expected) else 0, expected

def create_db(db_filename):
    schema = {
        'coppie': '''
            CREATE TABLE IF NOT EXISTS coppie (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER NOT NULL,
            entry_1_id INTEGER NOT NULL,
            entry_1_ac TEXT,
            entry_1_sent TEXT,
            entry_1_video TEXT,
            entry_2_id INTEGER NOT NULL,
            entry_2_ac TEXT,
            entry_2_sent TEXT,
            entry_2_video TEXT,
            different INTEGER,
            UNIQUE(row_id,entry_1_id, entry_2_id)
        )'''
        ,
        'quadruple': '''
            CREATE TABLE IF NOT EXISTS quadruple (
                id INTEGER PRIMARY KEY, 
                s1 TEXT, v1 TEXT, s2 TEXT, v2 TEXT,
                s3 TEXT, v3 TEXT, s4 TEXT, v4 TEXT, 
                intruder INTEGER
            )''',
        'coppie_evaluations': '''
            CREATE TABLE IF NOT EXISTS coppie_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                coppia_id INTEGER, prompt TEXT,
                model TEXT, answer TEXT, expected TEXT, accuracy INTEGER,
                prompt_type INTEGER, fewshot BOOLEAN,
                response_time DATETIME,
                FOREIGN KEY (coppia_id) REFERENCES coppie(id)
            )''',
        'quadruple_evaluations': '''
            CREATE TABLE IF NOT EXISTS quadruple_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                quadrupla_id INTEGER, prompt TEXT,
                model TEXT, answer TEXT, expected TEXT, accuracy INTEGER,
                prompt_type INTEGER, fewshot BOOLEAN,
                response_time DATETIME,
                FOREIGN KEY (quadrupla_id) REFERENCES quadruple(id)
            )'''
    }
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()
    for s in schema.values():
        cur.execute(s)
    conn.commit()
    conn.close()

def populate_db(db_filename):
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()

    quadruples = load_jsonl('../nuovi_jsonl/macid_dataset_huggingface_filtered.jsonl')
    cur.executemany("INSERT OR IGNORE INTO quadruple VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(q['id'], q['s1'], q['v1'], q['s2'], q['v2'], q['s3'], q['v3'], q['s4'], q['v4'], q['intruder']) for q in quadruples])

    pairs = load_jsonl('../nuovi_jsonl/macid_dataset_pairs_no_examples.jsonl')
    cur.executemany("INSERT OR IGNORE INTO coppie (row_id, entry_1_id, entry_1_ac, entry_1_sent, entry_1_video, entry_2_id, entry_2_ac, entry_2_sent, entry_2_video, different) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(p['row_id'], p['entry_1_id'], p['entry_1_ac'], p['entry_1_sent'], p['entry_1_video'], p['entry_2_id'], p['entry_2_ac'], p['entry_2_sent'], p['entry_2_video'], p['different']) for p in pairs])
    
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--processor", required=True, choices=['dummy', 'oai', 'cerebras'])
    parser.add_argument("--db-file", type=str, default="macid.db")
    parser.add_argument("--model", type=str)
    parser.add_argument("--api-key", type=str, default='EMPTY')
    parser.add_argument("--api-endpoint", type=str, default='http://localhost:8000/v1')
    parser.add_argument("--setup-db", action="store_true")
    parser.add_argument("--openrouter_noquant", action="store_true")
    parser.add_argument("--openrouter_noreasoning", action="store_true")

    args = parser.parse_args()

    create_db(args.db_file)
    populate_db(args.db_file)

    if args.setup_db:
        print("Database pronto.")
        exit(0)


    eval_macid(args.model, args.api_key, args.api_endpoint, args.db_file, args.openrouter_noquant, args.openrouter_noreasoning)
