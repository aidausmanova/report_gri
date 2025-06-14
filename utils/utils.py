import re
import nltk
import spacy
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

boilerplate_patterns = [
    r"we are committed to[^.]*\.",
    r"is committed to[^.]*\.",
    r"we continue to [^.]*\.",
    r"in line with our values[^.]*\.",
    r"as part of our continuous improvement[^.]*\.",
    r"our mission is to[^.]*\.",
    r"^this report.*?\.",
    r"^read more about [^.]*\.",
    r"^see more about [^.]*\.",
    r"^you can find [^.]*\.",
    r"^we strive to [^.]*\.",
]

def save_json(path, content):
    with open(path, 'w') as f:
        json.dump(content, f, indent=2)

def read_json(path):
    with open(path) as json_data:
        content = json.load(json_data)
    return content

def get_gri_standards(path):
    with open(path, "r") as file:
        gri_data = json.load(file)

    gri_standards_corpus = {}
    gri_list = []
    for gri in gri_data:
        gri_standards_corpus[gri['idx']] = f"{gri['title']} ({', '.join(gri['topics'])}). {gri['description']}"
        gri_list.append(f"{gri['idx']}: {gri['description']}")

    return gri_standards_corpus, gri_list

def get_gri_disclosures(path):
    with open(path, "r") as file:
        gri_data = json.load(file)

    gri_disclosures_corpus = {}
    gri_disclosures_list = []
    for gri in gri_data:
        disclosures = gri["disclosures"]
        for disclosure in disclosures:
            gri_disclosures_corpus[disclosure['idx']] = f"{disclosure['title']} ({disclosure['description']})"
            gri_disclosures_list.append(f"{disclosure['idx']}: {disclosure['description']}")
    return gri_disclosures_corpus, gri_disclosures_list

def get_section_passages(corpus):
    section_data = []
    cur_section = corpus[0]['idx'].split("_")[0]
    cur_title = corpus[0]['title']
    # section_docs = [f"{corpus[0]['title']}\n{corpus[0]['text']}"]
    section_corpus = {
        cur_section: f"{corpus[0]['title']}\n{corpus[0]['text']}"
    }
    section_data.append({"section_idx": cur_section, "title": cur_title, "text": corpus[0]['text'], "gri": []})

    for i in range (1, len(corpus)):
        chunk = corpus[i]
        section_id = chunk['idx'].split("_")[0]
        if chunk['idx'].startswith(cur_section):
            section_corpus[cur_section] += f"\n{chunk['text']}"
            section_data[-1]["text"] += f"\n{chunk['text']}" 
            # section_docs[-1] += f"\n{chunk['text']}"
        else:
            # section_docs.append(f"{chunk['title']}\n{chunk['text']}")
            cur_section = section_id
            cur_title = chunk['title']
            section_corpus[cur_section] = f"{chunk['title']}\n{chunk['text']}"
            section_data.append({"section_idx": cur_section, "title": cur_title, "text": chunk['text'], "gri": []})
        
    print("Section corpus length: ", len(section_corpus))
    return section_corpus, section_data


# def get_section_passages(corpus):
#     cur_section = corpus[0]['idx'].split("_")[0]
#     # section_docs = [f"{corpus[0]['title']}\n{corpus[0]['text']}"]
#     section_corpus = {
#         cur_section: f"{corpus[0]['title']}\n{corpus[0]['text']}"
#     }

#     for i in range (1, len(corpus)):
#         chunk = corpus[i]
#         section_id = chunk['idx'].split("_")[0]
#         if chunk['idx'].startswith(cur_section):
#             section_corpus[cur_section] += f"\n{chunk['text']}"
#             # section_docs[-1] += f"\n{chunk['text']}"
#         else:
#             # section_docs.append(f"{chunk['title']}\n{chunk['text']}")
#             cur_section = section_id
#             section_corpus[cur_section] = f"{chunk['title']}\n{chunk['text']}"
#     print("Section corpus length: ", len(section_corpus))
#     return section_corpus

def preprocess_paragraph(section_corpus):
    stop_words = set(stopwords.words('english'))
    max_sentences = 5
    new_section_corpus = section_corpus

    with open("data/taxonomies/gri_taxonomy_full_new.json", "r") as file:
        gri_data = json.load(file)

    disclosure_keywords = {}
    for gri in gri_data:
        disclosure_keywords[gri['idx']] = gri['topics']

    for section in tqdm(new_section_corpus):
        for pattern in boilerplate_patterns:
            paragraph = re.sub(pattern, '', section['text'], flags=re.IGNORECASE)
        sentences = sent_tokenize(paragraph)
        
        keywords = []
        for disclosure_keyword in disclosure_keywords.values():
            keywords.extend(disclosure_keyword)

        keyword_filtered = [
            s for s in sentences if any(kw.lower() in s.lower() for kw in keywords)
        ]

        cleaned_sentences = []

        for sent in keyword_filtered:
            words = word_tokenize(sent)
            non_stop = [w for w in words if w.lower() not in stop_words and w.isalnum()]
            if len(non_stop) >= 3:
                cleaned_sentences.append(sent)
        # print("Cleaned: ", len(cleaned_sentences))
        if not cleaned_sentences:
            cleaned_sentences = keyword_filtered
        
        if len(cleaned_sentences) <= max_sentences:
            section['text'] = " ".join(cleaned_sentences)
        else:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_sentences = [cleaned_sentences[i] for i in sorted(top_indices)]
            section['text'] = " ".join(top_sentences)
    return new_section_corpus

def process_llm_response(report_name):
    with open(f"output/{report_name}/gri_coverage_evaluation.json", "r") as f:
        data = json.load(f)

    completeness_pattern = re.compile(r'''(?i)completeness[\\]?[\s"']*[:=]?[\s"']*(?P<completeness>\d+)''', re.VERBOSE)
    materiality_pattern = re.compile(r'''(?i)materiality[\\]?[\s"']*[:=]?[\s"']*(?P<materiality>\d+)''', re.VERBOSE)

    results = []
    for row in data:
        output_text = row['response']
        responses = output_text.split("\n")
        completeness_score, materiality_score = None, None
        comment = ""
        
        completeness_match = completeness_pattern.search(row['response'])
        materiality_match = materiality_pattern.search(row['response'])
        
        if completeness_match:
            # print("Completeness:", completeness_match.group(0))
            output_text = re.sub(completeness_pattern, '', output_text)
            matches = re.findall(r'\d+', completeness_match.group(0))
            if completeness_score == None and matches:
                completeness_score = int(matches[0])
        if materiality_match:
            # print("Materiality:", materiality_match.group(0))
            output_text = re.sub(materiality_pattern, '', output_text)
            matches = re.findall(r'\d+', materiality_match.group(0))
            if materiality_score == None and matches:
                materiality_score = int(matches[0])

        comment += output_text.replace('comment', '').replace('Comment', '').replace('"', '').replace(':', '').replace(",\n", '').replace("\n", '').replace("  ", '').replace("{", '').replace("}", '').replace(",,", '').replace("```", '').replace("json", '')
        results.append(
            {
                'disclosure': row['disclosure'],
                'section_ids': row['section_ids'],
                'completeness': completeness_score if completeness_score else 0,
                'materiality': materiality_score if materiality_score else 0,
                'comment': comment
                
            }
        )
    
    with open(f"output/{report_name}/final_evaluation.json", "w") as f:
        json.dump(results, f)