import json
import numpy as np

def save_json(path, content):
    with open(path, 'w') as f:
        json.dump(content, f, indent=2)

def read_json(path):
    with open(path) as json_data:
        content = json.load(json_data)
    return content

def get_gri_indicators(path):
    with open(path, "r") as file:
        gri_data = json.load(file)

    gri_corpus = {}
    for gri in gri_data:
        topics = gri["topics"]
        indicators = gri["indicators"]
        
        for indicator in indicators:
            indicator_name = indicator['indicator'].lower().split(" ")
            gri_corpus["_".join(indicator_name)] = f"{gri['title']} ({', '.join(topics)}): {indicator['disclosure']} ({indicator['details']})"

    return gri_corpus

def get_section_passages(corpus):
    cur_section = corpus[0]['idx'].split("_")[0]
    # section_docs = [f"{corpus[0]['title']}\n{corpus[0]['text']}"]
    section_corpus = {
        cur_section: f"{corpus[0]['title']}\n{corpus[0]['text']}"
    }

    for i in range (1, len(corpus)):
        chunk = corpus[i]
        section_id = chunk['idx'].split("_")[0]
        if chunk['idx'].startswith(cur_section):
            section_corpus[cur_section] += f"\n{chunk['text']}"
            # section_docs[-1] += f"\n{chunk['text']}"
        else:
            # section_docs.append(f"{chunk['title']}\n{chunk['text']}")
            cur_section = section_id
            section_corpus[cur_section] = f"{chunk['title']}\n{chunk['text']}"
    print("Section corpus length: ", len(section_corpus))
    return section_corpus

def get_80th_percentile_val(gri_matches):
    gri_sim_scores = {}
    for match in gri_matches:
        match_ranks = []
        report_name = match['report_idx']
        for section in match['sections']:
            for ranks in section.values():
                for gri, score in ranks:
                    match_ranks.append(score)
        gri_sim_scores[report_name] = match_ranks
    all_scores = np.concatenate(list(gri_sim_scores.values()))
    return np.percentile(all_scores, 80)