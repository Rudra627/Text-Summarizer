# <h1>Text Summarizer</h1> 

A FastAPI-based web application for automatic text summarization using the T5 (Text-to-Text Transfer Transformer) model. This project provides both a REST API and an interactive web interface for summarizing long texts into concise summaries.

---

## Features

- **Automatic Text Summarization**: Uses the Google FLAN-T5 model for high-quality text summarization
- **REST API Endpoint**: POST requests for programmatic access to summarization
- **Web Interface**: User-friendly UI for interactive text summarization
- **Text Preprocessing**: Automatic cleaning of input text (removes HTML tags, excess whitespace, etc.)
- **Intelligent Length Detection**: Validates input to ensure meaningful summaries
- **Device Optimization**: Automatically detects and uses GPU (CUDA/MPS) or CPU for inference
- **Logging**: Comprehensive logging for debugging and monitoring
- **Auto-reload**: Development server with auto-reload on code changes

---

## Tech Stack

- **Backend**: FastAPI
- **ML Model**: Hugging Face Transformers (T5/FLAN-T5)
- **Frontend**: HTML + Jinja2 Templates
- **Server**: Uvicorn
- **Deep Learning**: PyTorch
- **Other**: Pydantic for data validation, Regex for text cleaning

---

## Project Structure

```
Text_Summrizer/
├── app.py                  # Main FastAPI application
├── index.html             # Web UI template
├── __pycache__/           # Python cache files
└── save_model/            # Pre-trained model directory
    ├── config.json        # Model configuration
    ├── generation_config.json
    └── model.safetensors  # Model weights
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Steps

1. **Navigate to the project directory**:
   ```bash
   cd d:\NLP-Project\Text_Summrizer
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   ..\Text_Summrizer_env\Scripts\activate
   
   # Or create a new one
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn transformers torch pydantic jinja2 pyyaml
   ```

---

## Running the Application

### Start the Server

```bash
# Navigate to project directory
cd d:\NLP-Project\Text_Summrizer

# Activate virtual environment
..\Text_Summrizer_env\Scripts\activate

# Run the application
uvicorn app:app --reload
```

**Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started Server process
INFO:     Application startup complete.
```

### Access the Application

- **Web Interface**: http://127.0.0.1:8000/
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative API Docs**: http://127.0.0.1:8000/redoc

---

## API Endpoints

### 1. Home Page (GET)
**Endpoint**: `GET /`
- Returns the HTML web interface
- No parameters required

**Response**: HTML page for interactive text summarization

---

### 2. Summarize Text (POST)
**Endpoint**: `POST /summarize/`

**Request**:
```json
{
  "dialogue": "Your long text to summarize here..."
}
```

**Response - Success**:
```json
{
  "summary": "The summarized text..."
}
```

**Response - Error**:
```json
{
  "summary": "Error: [error message]",
  "error": true
}
```

**Example using cURL**:
```bash
curl -X POST "http://127.0.0.1:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Artificial intelligence is transforming the world..."}'
```

**Example using Python**:
```python
import requests

url = "http://127.0.0.1:8000/summarize/"
data = {"dialogue": "Your text here..."}
response = requests.post(url, json=data)
print(response.json())
```

---

## How It Works

### Text Processing Pipeline

1. **Input Validation**: Checks if text is provided
2. **Text Cleaning**: 
   - Removes line breaks (`\r\n`)
   - Normalizes whitespace
   - Removes HTML tags
   - Converts to lowercase
3. **Length Check**: Ensures minimum 5 words for meaningful summarization
4. **Model Prefix**: Adds "summarize: " prefix (required by T5 model)
5. **Tokenization**: Converts text to model-compatible tokens
6. **Generation**: Uses the T5 model to generate summary with:
   - Max length: 150 tokens
   - Min length: 10 tokens
   - Beam search: 4 beams
   - Temperature: 0.7
7. **Decoding**: Converts output tokens back to readable text

---

## Model Details

- **Model Name**: Google FLAN-T5 Base
- **Model Type**: T5ForConditionalGeneration
- **Vocabulary Size**: 32,128 tokens
- **Architecture**: Encoder-Decoder Transformer
- **Training Task**: Multi-task fine-tuning on 1,000+ tasks
- **Pre-trained Weights**: Approximately 242MB

### Model Parameters
- `d_model`: 512 (embedding dimension)
- `num_layers`: 6 (encoder & decoder layers)
- `num_heads`: 8 (attention heads)
- `d_ff`: 2048 (feed-forward dimension)

---

## Device Support

The application automatically selects the best available device:

1. **MPS** (Metal Performance Shaders) - Apple Silicon
2. **CUDA** - NVIDIA GPUs
3. **CPU** - Fallback for standard processors

Check the console logs to see which device is being used:
```
INFO:app:Using CUDA device
# or
INFO:app:Using CPU device
```

---

## Error Handling

The application handles various error scenarios:

| Error | Cause | Solution |
|-------|-------|----------|
| "Text is too short" | Input < 5 words | Provide longer text |
| "Error loading model" | Model files missing | Check `save_model/` directory |
| "File not found" | `index.html` missing | Ensure HTML template exists |
| Empty summary | Model compatibility issue | Check model weights integrity |

---

## Logging

All activities are logged to the console with timestamps and severity levels:

```
INFO:app:Received summarize request with text length: 512
INFO:app:Cleaned text: artificial intelligence (ai) is transforming...
INFO:app:Tokenization done
INFO:app:Generation done
INFO:app:Summary generated: AI is revolutionizing various industries...
```

---

## Performance Optimization Tips

1. **GPU Usage**: Install CUDA for faster GPU inference
2. **Batch Processing**: Modify the app to process multiple texts simultaneously
3. **Caching**: Implement caching for repeated summarizations
4. **Model Quantization**: Use quantized models for faster inference on CPU
5. **Production Deployment**: Use Gunicorn with multiple workers

---

## Troubleshooting

### Issue: Model returns empty summaries
- **Solution**: Ensure tokenizer files are downloaded or use `google/flan-t5-base`

### Issue: Port 8000 already in use
- **Solution**: Change port with `uvicorn app:app --port 8001 --reload`

### Issue: Slow inference
- **Solution**: Install PyTorch with GPU support or reduce beam search size

### Issue: Module not found errors
- **Solution**: Ensure virtual environment is activated and dependencies installed

---

## Future Enhancements

- [ ] Support for multiple summarization models (BART, Pegasus, etc.)
- [ ] Custom summary length selection
- [ ] Batch API for multiple documents
- [ ] Model fine-tuning on custom datasets
- [ ] WebSocket support for real-time summarization
- [ ] Database integration for history tracking
- [ ] Authentication and rate limiting
- [ ] Multi-language support

---

## Dependencies

```
fastapi==0.135.2
uvicorn==0.27.0
transformers==4.40.0
torch==2.11.0
pydantic==2.12.5
jinja2==3.1.6
pyyaml==6.0.3
```

---

## License

This project is open source and available under the MIT License.

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review console logs for error messages
3. Verify model files are present in `save_model/`
4. Ensure all dependencies are installed

---

## Author Notes

- The model uses the "summarize: " prefix as required by T5 task-specific training
- Text is automatically lowercased for consistent model input
- Beam search (num_beams=4) improves output quality at the cost of speed
- Early stopping prevents generating overly long summaries

---

**Last Updated**: March 30, 2026
