import os
import sys
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils.api_utils import *
# from utils.prompt_utils import *
from utils.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_topN(df, n=2):
    # top3_df = df.groupby(['gri_standard', 'gri_disclosure'])[col].nlargest(3).reset_index()
    top3_df = df.groupby(['gri_standard', 'gri_disclosure']).head(n).reset_index()
    return top3_df

def paragraph2gri_matching(section_index_corpus, gri_data, model, tokenizer):
    msmarco_rankings = {}
    msmarco_ranking_rows = []

    passage_ids = list(section_index_corpus.keys())
    paragraphs = list(section_index_corpus.values())

    for gri_standard in tqdm(gri_data):
        for gri_disclosure in gri_standard['disclosures']:
            gri_description = gri_disclosure['title']+' '+gri_disclosure['description']
            gri_texts = [gri_description] * len(paragraphs)
            inputs = tokenizer(gri_texts, paragraphs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with torch.no_grad():
                scores = model(**inputs).logits.squeeze().tolist()

            scored_passages = list(zip(passage_ids, scores))
            scored_passages.sort(key=lambda x: x[1], reverse=True)

            msmarco_rankings[gri_disclosure['idx']] = scored_passages
            for passage_id, score in scored_passages:
                msmarco_ranking_rows.append(
                    {
                        'gri_standard': gri_standard['idx'],
                        'gri_disclosure': gri_disclosure['idx'],
                        'section_idx': passage_id,
                        'score': score
                    }
                )
    return msmarco_ranking_rows

def run_retrieval(report_name):
    # Prepapare report data
    corpus_path = f"/storage/usmanova/report_gri/data/reports/annotated/{report_name}_section_corpus.json"
    # corpus_path = f"output/{report_name}/{report_name}_corpus.json"
    with open(corpus_path, "r") as f:
        section_corpus = json.load(f)
    section_index_corpus = {}
    for passage in section_corpus:
        section_index_corpus[passage['section_idx']] = passage['text']

    # Prepare GRI data
    # gri_standards_corpus, gri_standards = get_gri_standards("data/taxonomies/gri_taxonomy_full_new.json")
    # gri_disclosure_corpus, gri_disclosures = get_gri_disclosures("data/taxonomies/gri_taxonomy_full_new.json")
    # gri_id_to_index, gri_s2d = {}, {}
    # for i, (key, val) in enumerate(gri_standards_corpus.items()):
    #     gri_id_to_index[key] = i

    with open("data/taxonomies/gri_taxonomy_full_new.json", "r") as file:
        gri_data = json.load(file)
    # gri_s2d = {}
    # for gri in gri_data:
    #     gri_s2d[gri['idx']] = gri['disclosure_codes']


    model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"
    rerank_tokenizer = AutoTokenizer.from_pretrained(model_name)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    rerank_model = rerank_model.to(device)

    matchings = paragraph2gri_matching(section_index_corpus, gri_data, rerank_model, rerank_tokenizer)
    similarity_df = pd.DataFrame(matchings)

    # topN_similarity_df = get_topN(similarity_df, 2)
    topN_similarity_df = similarity_df[similarity_df.score >= 0.2]

    data_dict = []
    # for gri_standard in topN_similarity_df['gri_standard'].unique():
    #     disclosures = list(set(topN_similarity_df[topN_similarity_df.gri_standard == gri_standard]['gri_disclosure'].tolist()))
    #     section_ids = list(set(topN_similarity_df[topN_similarity_df.gri_standard == gri_standard]['section_idx'].tolist()))
    #     data_dict.append({
    #         'gri_standard': gri_standard,
    #         'gri_disclosures': disclosures,
    #         'section_ids': section_ids
    #     })

    for gri_disclosure in topN_similarity_df['gri_disclosure'].unique():
        section_ids = list(topN_similarity_df[topN_similarity_df.gri_disclosure == gri_disclosure]['section_idx'].tolist())
        scores = list(set(topN_similarity_df[topN_similarity_df.gri_disclosure == gri_disclosure]['score'].tolist()))
        data_dict.append({
            'gri_disclosure': gri_disclosure,
            'section_ids': section_ids
        })


    if not os.path.exists(f'output/{report_name}/'):
        os.makedirs(f'output/{report_name}/')
    with open(f'output/{report_name}/top_retrieved_paragraphs.json', 'w') as f:
        json.dump(data_dict, f)

    # topN_similarity_df.to_json(f'output/{report_name}/top3_retrieved_paragraphs.json', orient='split', compression='infer', index=True)
    print(f"[INFO] retrieval output saved to output/{report_name}/top_retrieved_paragraphs.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--report", type=str)
    args = parser.parse_args()
    
    run_retrieval(args.report)