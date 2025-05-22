import sys
import time
from os import listdir
from os.path import isfile, join
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import get_gri_indicators, get_section_passages, get_80th_percentile_val

reports_list = {
 'westpac-climate-2023': 'Westpac 2023 climate report',
 'newlook-sustainability-2023': 'New Look sustainability report',
 'riotinto-climate-2023': 'Rio Tinto 2023 Climate Change Report',
 'hm-sustainability-2022': 'HM Group 2022 Sustainability Disclosure',
 'jpmorgan-climate-2022': 'JP Morgan 2022 Climate Report',
 'aeo-esg-2022': 'AEO 2022 ESG Report',
 'meta-sustainability-2023': 'Meta 2023 Sustainability Report',
 'suncor-sustainability-2023': 'Suncor 2023 Sustainability Report',
 'veolia-esg-2023': 'Veolia 2023 ESG Report',
 'starbucks-impact-2022': 'Starbucks 2022 Environmental & Social Impact Report',
 'mastercard-esg-2022': 'Mastercard 2022 ESG Report',
 'boeing-sustainability-2023': 'Boeing 2023 Sustainability Report',
 'bxp-esg-2022': 'BXP 2022 ESG Report',
 'hsbc-annual-2023': 'HSBC 2023 Annual Holdings and Accounts Report',
 'microsoft-sustainability-2022': 'Microsoft 2022 Environmental Sustainability Report',
 'at&t-sustainability-2022': 'AT&T 2022 Sustainability Summary',
 'ryanair-sustainability-2022': 'Ryanair 2022 Sustainability Report',
 'astrazeneca-sustainability-2023': 'AstraZeneca 2023 Sustainability Report',
 'mckinsey-esg-2022': 'McKinsey 2022 ESG Full Report',
 'paypal-global-2023': 'PayPal 2023 Global Impact Report',
 'ctreit-esg-2022': 'CT REIT 2022 ESG Report',
 'unioxford-sustainability-2022': 'University of Oxford 2022 Sustainability Report',
 'lloyds-sustainability-2023': 'Lloyds 2023 sustainability report',
 'sony-sustainability-2023': 'Sony 2023 Sustainability Report',
 'netflix-esg-2022': 'Netflix 2022 ESG Report',
 'qantas-sustainability-2023': 'Qantas 2023 Sustainability Report',
 'walmart-esg-2023': 'Walmart 2023 ESG Highlights',
 'woolworths-sustainability-2023': 'Woolworths Group 2023 Sustainability Report',
 'deloitte-global-2023': 'Deloitte 2023 Global Impact Report'}

industry_list = {
 'westpac-climate-2023': 'Finance',
 'newlook-sustainability-2023': 'Fashion',
 'riotinto-climate-2023': 'Mining',
 'hm-sustainability-2022': 'Fashion',
 'jpmorgan-climate-2022': 'Consultancy',
 'aeo-esg-2022': 'Fashion',
 'meta-sustainability-2023': 'Technology',
 'suncor-sustainability-2023': 'Energy',
 'veolia-esg-2023': 'Energy',
 'starbucks-impact-2022': 'Retail',
 'mastercard-esg-2022': 'Finance',
 'boeing-sustainability-2023': 'Aviation',
 'bxp-esg-2022': 'Real estate',
 'hsbc-annual-2023': 'Finance',
 'microsoft-sustainability-2022': 'Technology',
 'at&t-sustainability-2022': 'Telecommunication',
 'ryanair-sustainability-2022': 'Aviation',
 'astrazeneca-sustainability-2023': 'Pharmaceutical',
 'mckinsey-esg-2022': 'Consultancy',
 'paypal-global-2023': 'Finance',
 'ctreit-esg-2022': 'Real estate',
 'unioxford-sustainability-2022': 'Education',
 'lloyds-sustainability-2023': 'Finance',
 'sony-sustainability-2023': 'Technology',
 'netflix-esg-2022': 'Technology',
 'qantas-sustainability-2023': 'Aviation',
 'walmart-esg-2023': 'Retail',
 'woolworths-sustainability-2023': 'Retail',
 'deloitte-global-2023': 'Consultancy'}


def get_top_bm25(docs, gri_indicators, n=10):
    from rank_bm25 import BM25Okapi

    gri_ids = [key for key in gri_indicators.keys()]
    tokenized_indicators = [indicator_value.lower().split() for indicator_value in gri_indicators.values()]
    bm25 = BM25Okapi(tokenized_indicators)
    bm25_rankings = {}

    for passage_ind, passage_text in docs.items():
        bm25_scores = bm25.get_scores(passage_text.lower().split())  # BM25 scores for the passage

        gri_score_pairs = [(gri_ids[i], bm25_scores[i]) for i in range(len(bm25_scores))]
        top_n = sorted(gri_score_pairs, key=lambda x: x[1], reverse=True)[:n]

        # bm25_ranking = sorted(zip(gri_indicators, bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_rankings[passage_ind] = top_n
    
    return bm25_rankings

def bert_rerank(bert_model, section_docs, gri_indicators, bm25_rankings):
    gri_embeddings = {}
    for gri_key, gri_value in gri_indicators.items():
        gri_embeddings[gri_key] = bert_model.encode(gri_value)

    heatmap_matrix = np.full((len(section_docs), len(gri_indicators)), np.nan)
    new_rankings = {}

    for passage_ind, (passage_id, passage_text) in enumerate(section_docs.items()):
        passage_embedding = bert_model.encode(passage_text)

        top_gri_ids = [ranking[0] for ranking in bm25_rankings[passage_id]]
        indicator_embeddings = [gri_embeddings[gri_id] for gri_id in top_gri_ids]

        sim_scores = cosine_similarity([passage_embedding], indicator_embeddings)  # shape: (10,)
        rerank_arr = []
        rerank_idx = np.argsort(sim_scores, axis=1)[:, :][:, ::-1][0]
        for rank_id in rerank_idx:
            rerank_arr.append((bm25_rankings[passage_id][rank_id][0], sim_scores[0][rank_id].astype(float)))
        new_rankings[passage_id] = rerank_arr

        for i, gri_id in enumerate(top_gri_ids):
            j = gri_id_to_index[gri_id] 
            heatmap_matrix[passage_ind, j] = sim_scores[0][i]
    
    return heatmap_matrix, new_rankings

def msmarco_rerank(marco_model, section_docs, gri_indicators, bm25_rankings):
    heatmap_matrix = np.full((len(section_docs), len(gri_indicators)), np.nan)
    msmarco_rankings = {}

    for passage_ind, (passage_id, passage_text) in enumerate(section_docs.items()):
        top_gri_ids = [ranking[0] for ranking in bm25_rankings[passage_id]]
        bm25_ranked_input = [(passage_text, gri_indicators[gri_id]) for gri_id in top_gri_ids]

        marco_scores = marco_model.predict(bm25_ranked_input)
        final_rankings = sorted(zip([ind for ind in top_gri_ids], marco_scores.astype(float)), key=lambda x: x[1], reverse=True)
        msmarco_rankings[passage_id] = final_rankings

        for i, gri_id in enumerate(top_gri_ids):
            j = gri_id_to_index[gri_id] 
            heatmap_matrix[passage_ind, j] = final_rankings[i][1]

    return heatmap_matrix, msmarco_rankings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranker', type=str, default='bm25')
    parser.add_argument('--reranker', type=str) # bert, msmarco
    parser.add_argument('--topN', type=int, default=10)
    args = parser.parse_args()

    if args.reranker == 'bert':
        reranker_model = SentenceTransformer('all-MiniLM-L6-v2')
    elif args.reranker == 'msmarco':
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 

    # In this version we mapped with GRIs related to sustainability (gri_taxonomy_small), but may need to map with gri_taxonomy_full as well
    gri_indicators = get_gri_indicators("data/taxonomies/gri_taxonomy_small.json")
    gri_id_to_index = {}
    for i, (key, val) in enumerate(gri_indicators.items()):
        gri_id_to_index[key] = i

    all_org_gri_matches = []
    org_dri_match_records = []

    for file_name in tqdm(reports_list.keys()):
        corpus_path = f"data/reports/processed/{file_name}_corpus.json"
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        # Retrieve sections from the report
        section_docs = get_section_passages(corpus)

        # Statistical ranking top N
        if argparse.ranker == "bm25":
            bm25_rankings = get_top_bm25(section_docs, gri_indicators, argparse.topN)

        # Reranking top N similar GRI indicators
        if argparse.reranker == "bert":
            heatmap_matrix, new_rankings = bert_rerank(reranker_model, section_docs, gri_indicators, bm25_rankings)
        elif argparse.reranler == "msmarco":
            heatmap_matrix, new_rankings = msmarco_rerank(reranker_model, section_docs, gri_indicators, bm25_rankings)

        all_org_gri_matches.append({"report_idx": file_name, "report_name": reports_list[file_name], "industry": industry_list[file_name], "sections": new_rankings})
        # for new_rank in new_rankings:

    timestr = time.strftime("%Y_%m_%d-%H_%M")
    with open(f"org_gri_match_{timestr}.json", 'w') as f:
        json.dump(all_org_gri_matches, f)