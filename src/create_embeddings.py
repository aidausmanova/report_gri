import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from langchain.document_loaders import HuggingFaceDatasetLoader
from sentence_transformers import SentenceTransformer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_gri_indicators(path):
    with open(path, "r") as file:
        gri_data = json.load(file)

    gri_corpus = {}
    for gri in gri_data:
        # indicators = gri["indicators"]
        # for indicator in indicators:
            # indicator_name = indicator['indicator'].lower().split(" ")
            # gri_corpus["_".join(indicator_name)] = f"{gri['title']} ({', '.join(topics)}): {indicator['disclosure']} ({indicator['details']})"
        # gri_corpus[gri['idx']] = f"{gri['title']}({', '.join(gri["topics"])}): {gri['details']}"
        gri_corpus[gri['idx']] = f"{gri['title']} ({'', ''.join(gri['topics'])}). {gri['details']}"

    return gri_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    output_path = args.output_path
    model_name = args.model.split('/')[1]

    model = SentenceTransformer(args.model, device='cuda')
    # model.max_seq_length = 4096
    model.to(device)
    print(f"[INFO] Model loaded {args.model}")

    gri_indicators = get_gri_indicators("data/taxonomies/gri_taxonomy_high_level.json")
    gri_id_to_index = {}
    for i, (key, val) in enumerate(gri_indicators.items()):
        gri_id_to_index[key] = i
    
    print("[INFO] Start encoding")
    gri_embeddings = []
    for val in tqdm(gri_indicators.values()):
        desc_emb = model.encode(val, device='cuda')
        gri_embeddings.append(desc_emb)

    gri_metadata = [{'code': k, 'desc': v} for k, v in gri_id_to_index.items()]

    print("[INFO] Encoding finished")
    gri_embeddings = np.vstack(gri_embeddings)
    np.save(f"{output_path}/gri_embeddings_{model_name}.npy", gri_embeddings)
   
    with open(f"{output_path}/gri_embeddings_{model_name}_metadat.json", 'w') as f:
        json.dump(gri_metadata, f)

    print("[INFO] Embeddings saved")