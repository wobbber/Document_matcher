

## Overview

The Report Assembler is an advanced document processing tool designed to automatically extract and organize relevant content from a collection of base documents based on a predefined checklist. It leverages AI technologies to intelligently parse and categorize information across different document types.

## Features

-  Multi-document parsing (PDF and TXT support)
-  AI-powered content extraction using Gemini But firsti used the llama3.2 model with quantized model
-  Intelligent section-based content organization
-  Automated report package generation
-  Flexible and configurable through JSON checklist

## Prerequisites

python3.10 or +3.8
 
### Required Libraries
- langchain
- transformers
- google-generativeai or  llama-cpp/ctransformers
- gradio (optional, for web interface)

## Installation

1. Clone the repository
```bash
git clone https://your-repository-url.git
cd folfer name 
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```
3. In have app.py and appui.py run the appui.py that gradio based ui give the checklist path and base document path take the folder as wellf  format.

## Configuration

### Checklist JSON Structure
Create a `checklist.json` with the following structure:

```json
{
    "product": "Your Product Name",
    "market": "Target Market",
    "sections": [
        {
            "name": "Section Name",
            "extraction_instructions": "Specific extraction guidelines"
        }
    ]
}
```

### Gemini API Configuration
Replace the API key in `app.py`:
```python
genai.configure(api_key="YOUR_GOOGLE_GEMINI_API_KEY")
```

## Usage

### Command Line
```bash
python app.py
```

### Gradio Web Interface
```bash
python interface.py
```

## Project Structure
```
report-package-assembler/
│
├── app.py                  # Core report generation logic
├── interface.py            # Gradio web interface
├── checklist1.json         # Sample checklist configuration
├── base_documents/         # Input documents directory
└── output_reports/         # Generated report packages
```

## Workflow
1. Prepare base documents in `base_documents/`
2. Configure `checklist.json`
3. Run the script
4. Find generated reports in `output_reports/`

## Logging
The application provides detailed logging to track document processing and potential issues.

## Error Handling
- Supports PDF and TXT file formats
- Graceful handling of unsupported file types
- Comprehensive error logging

## Future Enhancements
- [ ] Support more document formats
- [ ] Advanced AI model selection
- [ ] Enhanced content extraction strategies

## Troubleshooting
- Ensure all dependencies are installed
- Verify Gemini API key
- Check base document file formats



```

