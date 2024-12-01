import streamlit as st
import os
import json
import logging
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2Tokenizer
import google.generativeai as genai

class ReportPackageAssemblerUI:
    def __init__(self):
        # Setup page configuration
        st.set_page_config(
            page_title="Report Package Assembler", 
            page_icon="📊", 
            layout="wide"
        )
        
        # Initialize session state variables
        if 'generated_package' not in st.session_state:
            st.session_state.generated_package = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _initialize_language_model(self, api_key=None):
        try:
            # Option 1: Try Streamlit secrets
            if not api_key:
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                except Exception as secrets_error:
                    st.warning(f"Secrets error: {secrets_error}")
                    api_key = None

            # Option 2: Use environment variable
            if not api_key:
                api_key = os.getenv('GEMINI_API_KEY')

            # Option 3: Hardcoded key (NOT RECOMMENDED FOR PRODUCTION)
            if not api_key:
                api_key = "AIzaSyDZUv0QDHHD9ytqyHLYyG2V139MbYEl6lM"  # Replace with your actual key

            if not api_key:
                st.error("No API key found. Please provide a Gemini API key.")
                return None

            # Configure and initialize Gemini model
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Verify model initialization
            test_response = model.generate_content("Hello, can you confirm you're working?")
            st.success("Gemini model initialized successfully!")
            
            return model
        
        except Exception as e:
            st.error(f"Detailed Error Initializing Gemini Model: {e}")
            st.error("Possible reasons:")
            st.error("1. Invalid API Key")
            st.error("2. Network Issues")
            st.error("3. API Access Restrictions")
            return None

    def load_documents(self, base_documents):
        documents = []
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        for file in base_documents:
            # Save uploaded file temporarily
            with open(os.path.join("temp", file.name), "wb") as f:
                f.write(file.getbuffer())
            
            filepath = os.path.join("temp", file.name)
            
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif file.name.endswith('.txt'):
                loader = TextLoader(filepath)
            else:
                st.warning(f"Unsupported file type: {file.name}. Skipping.")
                continue
            
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            documents.extend(splits)
        
        return documents

    def generate_report_package(self, checklist, base_documents):
        # Ensure directories exist
        os.makedirs("output_reports", exist_ok=True)

        # Read checklist
        try:
            checklist_data = json.load(checklist)
        except Exception as e:
            st.error(f"Error reading checklist: {e}")
            return None

        # Initialize model with explicit API key input option
        api_key_input = st.text_input("Enter Gemini API Key", type="password")
        model = self._initialize_language_model(api_key_input)
        
        if not model:
            st.error("Failed to initialize language model. Cannot proceed.")
            return None

        # Tokenizer for token management
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Load documents
        documents = self.load_documents(base_documents)

        # Generate package structure
        product = checklist_data['product']
        market = checklist_data['market']
        package_path = os.path.join("output_reports", f"{product}_{market}")
        os.makedirs(package_path, exist_ok=True)

        # Process sections
        processed_sections = []
        for section in checklist_data['sections']:
            # Extract content for the section
            all_docs_content = " ".join([doc.page_content for doc in documents])
            
            extraction_prompt = f"""
            Extract content related to "{section['name']}" from the following documents.
            Extraction Instructions: {section.get('extraction_instructions', 'No specific instructions provided')}
            
            Documents:
            {all_docs_content}
            
            Response Format:
            - Provide relevant content concisely.
            - If no relevant content is found, state: "No specific content available."
            """

            try:
                response = model.generate_content(extraction_prompt)
                section_content = response.text
                
                # Store processed section
                processed_sections.append({
                    'name': section['name'],
                    'content': section_content
                })

                # Save section content to file
                section_file = os.path.join(package_path, f"{section['name'].replace(' ', '_')}.txt")
                with open(section_file, 'w', encoding='utf-8') as f:
                    f.write(section_content)

            except Exception as e:
                st.error(f"Error processing section {section['name']}: {e}")

        # Save metadata
        metadata = {
            "product": product,
            "market": market,
            "sections": [section['name'] for section in processed_sections]
        }
        
        metadata_file = os.path.join(package_path, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return {
            'product': product,
            'market': market,
            'sections': processed_sections,
            'metadata_file': metadata_file,
            'package_path': package_path
        }

    def render_ui(self):
        st.title("📊 Report Package Assembler")

        # Sidebar for file uploads
        st.sidebar.header("Upload Documents")
        
        # Checklist upload
        checklist = st.sidebar.file_uploader(
            "Upload Checklist JSON", 
            type=['json'],
            help="Upload the checklist JSON file that defines the report structure"
        )

        # Base documents upload
        base_documents = st.sidebar.file_uploader(
            "Upload Base Documents", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True,
            help="Upload PDF or TXT documents to extract content from"
        )

        # Generate button
        if st.sidebar.button("Generate Report Package"):
            # Validate inputs
            if not checklist:
                st.sidebar.error("Please upload a checklist JSON file")
                return
            
            if not base_documents:
                st.sidebar.error("Please upload at least one base document")
                return

            # Show loading
            with st.spinner("Generating Report Package..."):
                # Generate the report package
                generated_package = self.generate_report_package(checklist, base_documents)
                
                # Store in session state
                st.session_state.generated_package = generated_package

        # Display generated package
        if st.session_state.generated_package:
            pkg = st.session_state.generated_package
            
            # Tabs for display
            tab1, tab2 = st.tabs(["Sections", "Metadata"])
            
            with tab1:
                st.header("Report Sections")
                for section in pkg['sections']:
                    with st.expander(section['name']):
                        st.write(section['content'])
            
            with tab2:
                st.header("Package Metadata")
                st.write(f"**Product:** {pkg['product']}")
                st.write(f"**Market:** {pkg['market']}")
                st.write("**Sections:**")
                for section in pkg['sections']:
                    st.write(f"- {section['name']}")
                
                # Download metadata
                with open(pkg['metadata_file'], 'r', encoding='utf-8') as f:
                    metadata_json = f.read()
                st.download_button(
                    label="Download Metadata",
                    data=metadata_json,
                    file_name="metadata.json",
                    mime="application/json"
                )

def main():
    app = ReportPackageAssemblerUI()
    app.render_ui()

if __name__ == "__main__":
    main()
