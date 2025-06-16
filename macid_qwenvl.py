import transformers
from IPython.display import Video
import os
from datasets import load_dataset
from transformers import Qwen2_5VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import csv
import sys
import tqdm

video_path="/home/matteo/Ricerca/MACID/macid_dataset/macid_videos/"
ds_path="/home/matteo/Ricerca/MACID/nuovi_jsonl/macid_dataset_pairs_no_examples.jsonl"
macid_ds=list()
with open(ds_path,"r") as f:
    for row in f:
        data=json.loads(row)
        macid_ds.append(data)

print(f"Caricate {len(macid_ds)} coppie.")
corretti=0
for i in macid_ds:
    if os.path.exists(video_path+i['entry_1_video']+'.mp4'):
        pass
    else:
        print(f"Video {i['entry_1_video']} MANCANTE!")
        sys.exit(1)
    if os.path.exists(video_path+i['entry_2_video']+'.mp4'):
        pass
    else:
        print(f"Video {i['entry_2_video']} MANCANTE!")
        sys.exit(1) 
print(f"Tutti i video sono correttamente presenti nella cartella {video_path}")

prompt="In questo task ti verranno proposte coppie di video che mostrano azioni fisiche. Le azioni nelle coppie possono essere dello stesso tipo, ovvero possono rappresentare lo stesso concetto azionale, oppure essere di due tipi diversi. Il tuo compito è di indicare se le seguenti coppie di video esprimono lo stesso concetto azionale oppure no. Un concetto azionale è un’entità linguistico-cognitiva corrispondente a un pattern di modifiche del mondo compiute da un agente, ed è generalizzabile a vari oggetti (o azioni). Un concetto azionale può essere realizzato linguisticamente con più verbi e, viceversa, un verbo può rappresentare più concetti azionali distinti. Rispondi 'Sì' se ritieni che entrambi i video si riferiscano allo stesso concetto azionale, rispondi 'No' se ritieni che mostrino due concetti azionali diversi."

model_id="Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
print(f"Modello {model_id} caricato.")
correct = 0
results = []

pbar = tqdm.tqdm(macid_ds)
for item in pbar:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}{item['entry_1_video']}.mp4",
                    "max_pixels": 720*576,
                    "fps": 2,
                },
                {
                    "type": "video", 
                    "video": f"file://{video_path}{item['entry_2_video']}.mp4",
                    "max_pixels": 720*576,
                    "fps": 2,
                },
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    result = output_text[0].lower()
    
    if int(item['different'])==1:
        target='n'
    else:
        target='s'
    
    is_correct = 0
    if (result[0]==target):
        is_correct = 1
        correct += 1
    
    results.append({
        'row_id': item.get('row_id', ''),
        'entry_1_id': item.get('entry_1_id', ''),
        'entry_2_id': item.get('entry_2_id', ''),
        'correct': is_correct
    })
    
    pbar.set_postfix({'Accuracy': f'{correct}/{pbar.n}={correct/max(1,pbar.n):.3f}'})

print(f"Accuracy: {correct}/{len(macid_ds)} = {correct/len(macid_ds):.3f}")

with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['row_id', 'entry_1_id', 'entry_2_id', 'correct']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
