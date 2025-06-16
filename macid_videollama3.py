import os
import json
import csv
import sys
import tqdm
import base64
import cv2
from PIL import Image
from IPython.display import Markdown, clear_output, display, Video
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor

model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16 
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


video_path = "/home/matteo/Ricerca/MACID/macid_videos/"
ds_path = "/home/matteo/Ricerca/MACID/nuovi_jsonl/macid_dataset_pairs_no_examples.jsonl"

macid_ds = []
with open(ds_path, "r") as f:
    for row in f:
        macid_ds.append(json.loads(row))
print(f"Caricate {len(macid_ds)} coppie.")
for item in macid_ds:
    for v in [item['entry_1_video'], item['entry_2_video']]:
        if not os.path.exists(video_path + v + '.mp4'):
            print(f"Video {v} MANCANTE!")
            sys.exit(1)
print(f"Tutti i video sono correttamente presenti nella cartella {video_path}")


prompt="In questo task ti verranno proposte coppie di video che mostrano azioni fisiche. Le azioni nelle coppie possono essere dello stesso tipo, ovvero possono rappresentare lo stesso concetto azionale, oppure essere di due tipi diversi. Il tuo compito è di indicare se le seguenti coppie di video esprimono lo stesso concetto azionale oppure no. Un concetto azionale è un’entità linguistico-cognitiva corrispondente a un pattern di modifiche del mondo compiute da un agente, ed è generalizzabile a vari oggetti (o azioni). Un concetto azionale può essere realizzato linguisticamente con più verbi e, viceversa, un verbo può rappresentare più concetti azionali distinti. Rispondi 'Sì' se ritieni che entrambi i video si riferiscano allo stesso concetto azionale, rispondi 'No' se ritieni che mostrino due concetti azionali diversi."

correct = 0
results = []
pbar = tqdm.tqdm(macid_ds)

for item in pbar:
    v1_path = os.path.join(video_path, item['entry_1_video'] + '.mp4')
    v2_path = os.path.join(video_path, item['entry_2_video'] + '.mp4')
    conversation = [
        {        
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": {"video_path": v1_path, "fps": 8, "max_frames": 180}
                },
                {
                    "type": "video", 
                    "video": {"video_path": v2_path, "fps": 8, "max_frames": 180}
                },            
                {
                    "type": "text", 
                    "text": prompt
                },
            ]
        }
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=64)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


    result_clean = response.lower().strip()
    target = 'n' if int(item['different']) == 1 else 's'
    is_correct = int(result_clean.startswith(target))
    correct += is_correct

    result = {
        'row_id': item.get('row_id', ''),
        'entry_1_id': item.get('entry_1_id', ''),
        'entry_2_id': item.get('entry_2_id', ''),
        'raw_output': output_text,
        'correct': is_correct
    }

    with open("risultati_video_vllama3.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    results.append(result)
    pbar.set_postfix({'Accuracy': f'{correct}/{pbar.n} = {correct / max(1, pbar.n):.3f}'})

print(f"Accuracy: {correct}/{len(macid_ds)} = {correct/len(macid_ds):.3f}")

with open('results1.csv', 'w', newline='') as csvfile:
    fieldnames = ['row_id', 'entry_1_id', 'entry_2_id', 'raw_output', 'correct']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
