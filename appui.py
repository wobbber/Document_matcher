import os
import json
import gradio as gr
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import ReportPackageAssembler

def generate_report_package(checklist_path, base_documents_path, output_path):
    try:
        # Ensure paths are absolute
        checklist_path = os.path.abspath(checklist_path)
        base_documents_path = os.path.abspath(base_documents_path)
        output_path = os.path.abspath(output_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Initialize and run the assembler
        assembler = ReportPackageAssembler(
            checklist_path=checklist_path,
            base_documents_path=base_documents_path,
            output_path=output_path
        )
        assembler.generate_report_package()

        # Read the metadata file
        product = assembler.checklist['product']
        market = assembler.checklist['market']
        package_path = os.path.join(output_path, f"{product}_{market}")
        metadata_file = os.path.join(package_path, 'metadata.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Prepare metadata display
        metadata_str = json.dumps(metadata, indent=2)
        
        # List section folders
        section_folders = [
            os.path.join(package_path, section.replace(' ', '_')) 
            for section in metadata['sections']
        ]

        return (
            f"Report package generated successfully in: {package_path}\n\n" 
            "Metadata Contents:\n" + 
            metadata_str, 
            "\n".join(section_folders)
        )

    except Exception as e:
        return f"Error: {str(e)}", ""

def read_section_content(section_folder):
    try:
        content_file = os.path.join(section_folder, 'section_content.txt')
        with open(content_file, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading section content: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Report Package Assembler")
    
    with gr.Row():
        checklist_input = gr.Textbox(label="Checklist Path", placeholder="Enter path to checklist.json")
        base_docs_input = gr.Textbox(label="Base Documents Path", placeholder="Enter path to base documents directory")
        output_path_input = gr.Textbox(label="Output Path", placeholder="Enter path for output reports")
    
    generate_btn = gr.Button("Generate Report Package")
    
    metadata_output = gr.Textbox(label="Metadata", interactive=False, lines=10)
    section_folders = gr.Dropdown(label="Section Folders")
    
    section_content = gr.Textbox(label="Section Content", interactive=False, lines=10)

    # Generate Report Package
    generate_btn.click(
        fn=generate_report_package, 
        inputs=[checklist_input, base_docs_input, output_path_input],
        outputs=[metadata_output, section_folders]
    )

    # Read Section Content
    section_folders.change(
        fn=read_section_content,
        inputs=[section_folders],
        outputs=[section_content]
    )

    # Example values for convenience
    demo.load(
        fn=lambda: (
            "checklist1.json", 
            "base_document", 
            "output_reports"
        ),
        outputs=[checklist_input, base_docs_input, output_path_input]
    )

if __name__ == "__main__":
    demo.launch()
