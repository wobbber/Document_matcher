# import os
# import json
# from typing import List, Dict
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import LlamaCpp
# from transformers import GPT2Tokenizer
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# class ReportPackageAssembler:
#     def __init__(self, checklist_path: str, base_documents_path: str, output_path: str = 'output'):
#         """
#         Initialize Report Package Assembler with LangChain and LlamaCpp.
        
#         Args:
#             checklist_path (str): Path to checklist JSON file.
#             base_documents_path (str): Path to directory containing base documents.
#             output_path (str): Path to save generated report package.
#         """
#         # Load checklist
#         with open(checklist_path, 'r') as f:
#             self.checklist = json.load(f)
        
#         self.base_documents_path = base_documents_path
#         self.output_path = output_path
        
#         # Initialize LlamaCpp model
#         self._initialize_language_model()

#         # Initialize GPT-2 tokenizer for token counting
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#     def _initialize_language_model(self):
#         """
#         Initialize LlamaCpp model for text processing.
#         """
#         try:
#             model_path = "Llama-3.2-3B-Instruct-IQ3_M.gguf"  # Update this with your model path
#             self.llm = LlamaCpp(
#                 model_path=model_path,
#                 n_gpu_layers=1,
#                 n_batch=512,
#                 max_tokens=512,
#                 f16_kv=True,
#                 temperature=0.7
#             )
#             logging.info("LlamaCpp model initialized successfully.")
#         except Exception as e:
#             logging.error(f"Error initializing LlamaCpp model: {e}")
#             raise

#     def load_documents(self) -> List[Dict]:
#         """
#         Load and process base documents from the specified directory.
        
#         Returns:
#             List of processed documents.
#         """
#         documents = []
#         for filename in os.listdir(self.base_documents_path):
#             filepath = os.path.join(self.base_documents_path, filename)
            
#             if filename.endswith('.pdf'):
#                 loader = PyPDFLoader(filepath)
#             elif filename.endswith('.txt'):
#                 loader = TextLoader(filepath)
#             else:
#                 logging.warning(f"Unsupported file type: {filename}. Skipping.")
#                 continue
            
#             docs = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             splits = text_splitter.split_documents(docs)
#             documents.extend(splits)
        
#         return documents

#     def count_tokens(self, text: str) -> int:
#         """
#         Count the number of tokens in the provided text using GPT-2 tokenizer.
        
#         Args:
#             text (str): The input text for tokenization.
        
#         Returns:
#             int: The token count.
#         """
#         return len(self.tokenizer.encode(text))

#     def extract_section_content(self, documents: List[Dict], section: Dict) -> str:
#         """
#         Extract content for a specific section using the LlamaCpp model.
        
#         Args:
#             documents (List[Dict]): List of processed documents.
#             section (Dict): Section details from the checklist.
        
#         Returns:
#             str: Extracted content for the section.
#         """
#         # Combine all document contents
#         all_docs_content = " ".join([doc.page_content for doc in documents])
        
#         # Split the combined document content into chunks that do not exceed the token limit
#         content_chunks = self._split_content_by_tokens(all_docs_content)

#         # Create extraction prompt for each chunk
#         extracted_content = ""
#         for chunk in content_chunks:
#             extraction_prompt = f"""
#             Extract content related to "{section['name']}" from the following documents.
#             Extraction Instructions: {section.get('extraction_instructions', 'No specific instructions provided')}
            
#             Documents:
#             {chunk}
            
#             Response Format:
#             - Provide relevant content concisely.
#             - If no relevant content is found, state: "No specific content available."
#             """

#             try:
#                 extracted_content += self.llm(extraction_prompt) + "\n"
#             except Exception as e:
#                 logging.error(f"Error extracting content for section '{section['name']}': {e}")
#                 extracted_content += "Error occurred during content extraction.\n"
        
#         return extracted_content

#     def _split_content_by_tokens(self, content: str) -> List[str]:
#         """
#         Split the content into chunks that do not exceed the token limit of 512 tokens.
        
#         Args:
#             content (str): The content to be split into chunks.
        
#         Returns:
#             List[str]: A list of text chunks.
#         """
#         max_tokens = 512
#         tokens = self.tokenizer.encode(content)
        
#         chunks = []
#         for i in range(0, len(tokens), max_tokens):
#             chunk_tokens = tokens[i:i + max_tokens]
#             chunk = self.tokenizer.decode(chunk_tokens)
#             chunks.append(chunk)
        
#         return chunks

#     def generate_report_package(self):
#         """
#         Generate the complete report package based on the checklist and base documents.
#         """
#         # Create output directory
#         product = self.checklist['product']
#         market = self.checklist['market']
#         package_path = os.path.join(self.output_path, f"{product}_{market}")
#         os.makedirs(package_path, exist_ok=True)
#         logging.info(f"Report package directory created: {package_path}")

#         # Load base documents
#         documents = self.load_documents()

#         # Process each section
#         for section in self.checklist['sections']:
#             section_dir = os.path.join(package_path, section['name'].replace(' ', '_'))
#             os.makedirs(section_dir, exist_ok=True)
            
#             # Extract content for the section
#             content = self.extract_section_content(documents, section)
            
#             # Save content to a file
#             section_file = os.path.join(section_dir, 'section_content.txt')
#             with open(section_file, 'w') as f:
#                 f.write(content)
#             logging.info(f"Section content saved: {section_file}")

#         # Save metadata
#         metadata = {
#             "product": product,
#             "market": market,
#             "sections": [section['name'] for section in self.checklist['sections']]
#         }
#         metadata_file = os.path.join(package_path, 'metadata.json')
#         with open(metadata_file, 'w') as f:
#             json.dump(metadata, f, indent=2)
#         logging.info(f"Metadata saved: {metadata_file}")


# def main():
#     """
#     Entry point for the Report Package Assembler.
#     """
#     assembler = ReportPackageAssembler(
#         checklist_path='checklist.json',
#         base_documents_path='base_document',
#         output_path='output_reports'
#     )
#     assembler.generate_report_package()


# if __name__ == "__main__":
#     main()

import os
import json
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2Tokenizer
import google.generativeai as genai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ReportPackageAssembler:
    def __init__(self, checklist_path: str, base_documents_path: str, output_path: str = 'output'):
        """
        Initialize Report Package Assembler with LangChain and Gemini AI.
        
        Args:
            checklist_path (str): Path to checklist JSON file.
            base_documents_path (str): Path to directory containing base documents.
            output_path (str): Path to save generated report package.
        """
        # Load checklist
        with open(checklist_path, 'r') as f:
            self.checklist = json.load(f)
        
        self.base_documents_path = base_documents_path
        self.output_path = output_path
        
        # Initialize Gemini model
        self._initialize_language_model()

        # Initialize GPT-2 tokenizer for token counting
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def _initialize_language_model(self):
        """
        Initialize Gemini model for text processing.
        """
        try:
            # Configure the API key
            genai.configure(api_key="AIzaSyCtmG9Ho8eUPgwkRF9s7iU_z-S0r07CsIw")  # Replace with your actual API key

            # Initialize the Gemini model
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logging.info("Gemini model initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Gemini model: {e}")
            raise

        # Uncomment this if you want to use LlamaCpp instead of Gemini:
        # try:
        #     model_path = "Llama-3.2-3B-Instruct-IQ3_M.gguf"  # Update this with your model path
        #     self.llm = LlamaCpp(
        #         model_path=model_path,
        #         n_gpu_layers=1,
        #         n_batch=512,
        #         max_tokens=512,
        #         f16_kv=True,
        #         temperature=0.7
        #     )
        #     logging.info("LlamaCpp model initialized successfully.")
        # except Exception as e:
        #     logging.error(f"Error initializing LlamaCpp model: {e}")
        #     raise

    def load_documents(self) -> List[Dict]:
        """
        Load and process base documents from the specified directory.
        
        Returns:
            List of processed documents.
        """
        documents = []
        for filename in os.listdir(self.base_documents_path):
            filepath = os.path.join(self.base_documents_path, filename)
            
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif filename.endswith('.txt'):
                loader = TextLoader(filepath)
            else:
                logging.warning(f"Unsupported file type: {filename}. Skipping.")
                continue
            
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            documents.extend(splits)
        
        return documents

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the provided text using GPT-2 tokenizer.
        
        Args:
            text (str): The input text for tokenization.
        
        Returns:
            int: The token count.
        """
        return len(self.tokenizer.encode(text))

    def extract_section_content(self, documents: List[Dict], section: Dict) -> str:
        """
        Extract content for a specific section using the Gemini model.
        
        Args:
            documents (List[Dict]): List of processed documents.
            section (Dict): Section details from the checklist.
        
        Returns:
            str: Extracted content for the section.
        """
        # Combine all document contents
        all_docs_content = " ".join([doc.page_content for doc in documents])
        
        # Split the combined document content into chunks that do not exceed the token limit
        content_chunks = self._split_content_by_tokens(all_docs_content)

        # Create extraction prompt for each chunk
        extracted_content = ""
        for chunk in content_chunks:
            extraction_prompt = f"""
            Extract content related to "{section['name']}" from the following documents.
            Extraction Instructions: {section.get('extraction_instructions', 'No specific instructions provided')}
            
            Documents:
            {chunk}
            
            Response Format:
            - Provide relevant content concisely.
            - If no relevant content is found, state: "No specific content available."
            """

            try:
                response = self.model.generate_content(extraction_prompt)
                extracted_content += response.text + "\n"
            except Exception as e:
                logging.error(f"Error extracting content for section '{section['name']}': {e}")
                extracted_content += "Error occurred during content extraction.\n"
        
        return extracted_content

    def _split_content_by_tokens(self, content: str) -> List[str]:
        """
        Split the content into chunks that do not exceed the token limit of 512 tokens.
        
        Args:
            content (str): The content to be split into chunks.
        
        Returns:
            List[str]: A list of text chunks.
        """
        max_tokens = 512
        tokens = self.tokenizer.encode(content)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        
        return chunks

    def generate_report_package(self):
        """
        Generate the complete report package based on the checklist and base documents.
        """
        # Create output directory
        product = self.checklist['product']
        market = self.checklist['market']
        package_path = os.path.join(self.output_path, f"{product}_{market}")
        os.makedirs(package_path, exist_ok=True)
        logging.info(f"Report package directory created: {package_path}")

        # Load base documents
        documents = self.load_documents()

        # Process each section
        for section in self.checklist['sections']:
            section_dir = os.path.join(package_path, section['name'].replace(' ', '_'))
            os.makedirs(section_dir, exist_ok=True)
            
            # Extract content for the section
            content = self.extract_section_content(documents, section)
            
            # Save content to a file
            section_file = os.path.join(section_dir, 'section_content.txt')
            with open(section_file, 'w') as f:
                f.write(content)
            logging.info(f"Section content saved: {section_file}")

        # Save metadata
        metadata = {
            "product": product,
            "market": market,
            "sections": [section['name'] for section in self.checklist['sections']]
        }
        metadata_file = os.path.join(package_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Metadata saved: {metadata_file}")


def main():
    """
    Entry point for the Report Package Assembler.
    """
    assembler = ReportPackageAssembler(
        checklist_path='checklist1.json',
        base_documents_path='base_document',
        output_path='output_reports'
    )
    assembler.generate_report_package()


if __name__ == "__main__":
    main()
