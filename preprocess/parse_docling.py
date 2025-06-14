import warnings
# warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(os.getcwd())
import re
import time
import argparse
import logging
from pathlib import Path
from utils.utils import save_json

import docling
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions
)

_log = logging.getLogger(__name__)

def clean_text(text, rgx_list=[]):
    rgx_list = ["<!-- image -->", "; ->", "&gt", "amp;"]
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, '', new_text)
    return new_text

def parse(report_file):
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path(f"data/reports/original/{report_file}.pdf")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = False    

    pipeline_options.ocr_options.lang = ["en"]
    # pipeline_options.ocr_options.download_enabled = False
    # pipeline_options.ocr_options.model_storage_directory = "docling/models/EasyOcr"

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    doc_filename = conv_result.input.file.stem
    output_dir = Path("data/reports/processed/"+doc_filename)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        text = conv_result.document.export_to_text()
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    # with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
    #     fp.write(conv_result.document.export_to_markdown())
    
    return text

def download_models():
    docling.utils.model_downloader.download_models(
        output_dir = Path("docling/models")
    )
    print('models downloaded successfully.')

def run_parser(report_file):
    print(f"[INFO] Parsing {report_file}")
    text = parse(report_file)

    print("[INFO] Text preprocessing ...")
    p = re.compile("##.*\n")
    section_titles = re.findall(p, text)
    mod_section_titles = []
    for title in section_titles:
        mod_section_titles.append(title[3:len(title)-1])

    sections = text.split("##")
    sections.pop(0)
    print(f"[INFO] # sections: {len(sections)}, # titles: {len(mod_section_titles)}")
    # assert(len(sections) == len(mod_section_titles))

    file_name = "-".join(report_file.lower().split())
    chunked_paragraphs = []
    for idx, (section, title) in enumerate(zip(sections[:len(mod_section_titles)], mod_section_titles)):
        # Append paragraph data
        sec_text = section[len(title)+2:]
        sec_text = sec_text.split(". ")
        if len(sec_text) > 2:
            sec_text = ".".join(sec_text).replace('\n', '').replace('\u25cf', '')
            if not any(row['title'] == title for row in chunked_paragraphs):
                chunked_paragraphs.append({
                    "title": title,
                    "text": sec_text,
                    "section_idx": file_name+"-"+str(idx)
                })
            else:
                for row in chunked_paragraphs:
                    if row['title'] == title:
                        row['text'] += sec_text

    if not os.path.exists(f'output/{file_name}/'):
        os.makedirs(f'output/{file_name}/')
    output_dir = Path("output/"+file_name)
    save_json(output_dir / f"{file_name}_corpus.json", chunked_paragraphs)
    print(f"[INFO] Saving processed report to {output_dir}/{file_name}_corpus.json")
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--report", type=str)

    args = parser.parse_args()
    report_file = args.report

    run_parser(report_file)