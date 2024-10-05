import os
import re
import fitz
import torch
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English
from transformers import BitsAndBytesConfig
from sentencetransformer import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


class Preprocessing:
    """
    Take documents of different types and convert them to vector database
    
    Args: 
        doc_dir: Document directory (If the directory contains multiple files, handle that or if it contains only one file, handle that as well.)
        Returns: A Pandas dataframe of the preprocessed dataset
    """
    def __init__(self, doc_dir:os.path) -> None:
        pass

    def download_docs(self, url):
        """
        Takes in a url, downloads and save a document.
        """
        response = requests.get(url=url)
        filename = "doc"

        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"[INFO] The file has been downloaded and saved as {filename}")
        else:
            print(f"[INFO] Failed to download the file. Status Code: {response.status_code}")


    def open_and_read_pdf(self, pdf_path:str) -> list[str]:
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc)):
            text = page.get_text()
            text = self.format_text(text=text)
            pages_and_texts.append({
                "page_number": page_number,
                "page_char_count":len(text),
                "page_word_count":len(text.split(" ")), # I need to take care of special characters that do not count as words.
                "page_sentence_count_raw": len(text.split(". ")), # I need to include ?
                "page_token_count": len(text)/4, 
                "text":text
            })

        return pages_and_texts # Return a dataframe
    
    def chunk(self, pages_and_texts:list[str], chunk_size:int, filter_by_num_tokens:bool=False, min_token_length:int=30):

        # Create a spacy sentencizer
        sentencizer = English()
        sentencizer.add_pipe("sentencizer")

        # Sentencize the pages (break the page into a list of sentences)
        for item in tqdm(pages_and_texts):
            item["sentences"] = list(sentencizer(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["num_sentences_in_page"] = len(item["sentences"])

        # Loop through pages and texts and split sentences into chunks 
        for item in tqdm(pages_and_texts):
            item["sentence_chunks"] = self.split_list(input_list=item["sentences"], slice_size=chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        # Create a new data dictionary for the dataset
        pages_and_chunks = []
        for item in tqdm(pages_and_texts):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences in each chunk into a 'paragraph'
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Add some statistics to the chunks for filtering later
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk)/4 # 1 token = ~4chars
                
                pages_and_chunks.append(chunk_dict)
        
        if filter_by_num_tokens == True:
            return [item for item in pages_and_chunks if item["chunk_token_count"] >= min_token_length]
        else:
            return pages_and_chunks


    # Utility functions
    
    def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
        """
        Create list of slice_size for the sentences in a page.
        """
        return [input_list[i:i + slice_size] for i in range (0, len(input_list), slice_size)]


    def format_text(self, text:str) -> str:
        """
        Performs formatting on text. 
        I have to expand this to do more on the text
        """

        cleaned_text = text.replace("\n", " ").strip()

        return cleaned_text
    

    

class VectorDatabase:
    """
    Create a vector database data

    Args: 
        data: Pandas dataframe.
        Return: An indexed vector database
    """
    def __init__(self, data:list[dict]) -> pd.DataFrame:
        pass

    def create_embeddings(self, data:list[dict], model_name:str = "all-mpnet-base-v2", file_path:str = None, save_data:bool=False):
    
        # create embedding model
        embedding_model = SentenceTransformer(model_name_or_path = model_name, device="cuda")
        embedding_model.to("cuda")

        # create embeddings 
        text_chunks = [item["sentence_chunk"] for item in data] 
        embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)

        for i, item in enumerate(data):
            item["embedding"] = embeddings[i].cpu().numpy()

        df = pd.DataFrame(data)
        if save_data:
            assert file_path != None, "Please specify file path"
            df.to_csv(file_path, index=False)

        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        embeddings = torch.tensor(np.stack(df["embedding"].tolist(), axis=0), dtype=torch.float32)
        df = df.to_dict(orient="records")

        return df, embeddings
    

class SemanticSearch:
    """
    Process the documents, add them to the database and perform semantic search to find 

    Args:
        data: dictionary (directory for the documents)
        Return: return 
    """

    def __init__(self, dictionary) -> None:
        self.dictionary = dictionary # Add the dictionary

    
    def model(model_name:str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Create and Quantize model.
        """

        # Create a quantization configurations
        use_quantization_config = True
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) # load in 4 bit compute in float16

        # Falsh attention 2 = faster attention mechanism
        if (is_flash_attn_2_available()) and torch.cuda.get_device_capability(0)[0] >= 8:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa" #scaled dot product attention

        # Instantiate the model
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                        torch_dtype=torch.float16,
                                                        quantization_config=quantization_config if use_quantization_config else None,
                                                        low_cpu_mem_usage=True,
                                                        device_map="auto",
                                                        attn_implementation=attn_implementation)

        # send it to device
        if not use_quantization_config:
            llm_model.to("cuda")

        return llm_model

    def search(self, query):
        pass