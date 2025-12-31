# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChatCFD is an LLM-driven agent system that automates CFD (Computational Fluid Dynamics) simulations using OpenFOAM. It translates natural language descriptions or academic papers into executable OpenFOAM case configurations through a conversational interface.

## Build and Run Commands

### Environment Setup
```bash
# Create environment with conda
conda env create -f chatcfd_env.yml
conda activate chatcfd

# Alternative: pip install
pip install -r requirements.txt

# Download required sentence transformer model
python test_env/download_model.py
```

### Running the Application

**Web Interface (Streamlit):**
```bash
streamlit run src/chatbot.py --server.port 8501
```

**Command Line Mode:**
```bash
python src/chatbot.py --case_description_path <path_to_pdf_or_txt> --grid_path <path_to_mesh> --run_time <iterations>
```

### Verify Installation
```bash
# Check FAISS
python -c "import faiss; print(faiss.IndexFlatL2(10))"

# Check PyFoam
python -c "from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile; print('PyFoam OK')"
```

## Architecture

### Core Workflow
1. **PDF/Text Processing** (`chatbot.py`) - Extracts CFD case descriptions from uploaded documents
2. **Case Identification** - LLM identifies solver, turbulence model, and boundary conditions
3. **File Generation** (`file_preparation.py`) - Creates OpenFOAM case files (0/, constant/, system/)
4. **Execution** (`run_of_case.py`) - Runs OpenFOAM solver via subprocess
5. **Error Correction** (`file_corrector.py`, `Reflextion.py`) - Iterative error detection and fixing using LLM

### Key Modules

- **`config.py`** - Global configuration management using dataclasses; loads settings from `inputs/chatcfd_config.json`. Manages paths, LLM configs, case state, and OpenFOAM environment variables.

- **`qa_modules.py`** - LLM interface wrappers for DeepSeek V3 and R1 models. Classes: `QA_Context_deepseek_V3/R1` (with conversation history), `QA_NoContext_deepseek_V3/R1` (single query). `GlobalLogManager` tracks token usage.

- **`file_preparation.py`** - Generates OpenFOAM case files. Key functions:
  - `case_required_files()` - Determines required files based on solver/turbulence model
  - `generate_initial_files()` - Creates 0/, constant/, system/ files via LLM
  - `convert_mesh()` - Converts Fluent .msh to OpenFOAM polyMesh format
  - `extract_boundary_names()` - Parses mesh boundary conditions

- **`file_corrector.py`** - Error analysis and correction:
  - `analyse_error()` - Identifies files causing OpenFOAM runtime errors
  - `correct_error()` - Generates and applies fixes using LLM with tutorial references
  - `find_reference_files()` - Retrieves relevant OpenFOAM tutorial examples from database

- **`Reflextion.py`** - Implements reflection mechanism when repeated errors occur, helping LLM learn from failed correction attempts

- **`main_run_chatcfd.py`** - Main execution loop: generates files, runs simulation, performs iterative error correction (max 30 rounds)

- **`pdf_chunk_ask_question.py`** - RAG-based PDF processing using sentence-transformers for semantic search

### Data Flow
```
User Input (PDF/text + mesh)
    -> chatbot.py (UI/CLI entry point)
    -> config.py (load settings, OpenFOAM env)
    -> file_preparation.py (generate OF case files)
    -> run_of_case.py (execute simulation)
    -> file_corrector.py (if error, fix and retry)
    -> Reflextion.py (if repeated errors, reflect and adjust strategy)
```

### Database Directory (`database_OFv24/`)
Contains pre-processed OpenFOAM v2406 knowledge:
- `processed_merged_OF_cases.json` - Extracted tutorial case configurations
- `final_OF_solver_required_files.json` - Files required per solver
- `final_OF_turbulence_required_files.json` - Files required per turbulence model
- `OF_bc_entry.json` - Boundary condition entries
- `OF_case_dimensions.json` - Physical field dimensions

## Configuration

Edit `inputs/chatcfd_config.json` for:
- DeepSeek API keys and endpoints (V3 and R1 models)
- OpenFOAM installation path (default: `/usr/lib/openfoam/openfoam2406`)
- Sentence transformer model path
- Temperature settings for LLM calls

## Important Conventions

- OpenFOAM v2406 is required; other versions may cause database incompatibility
- Python 3.11 recommended; avoid 3.12 due to PyFoam conflicts
- Mesh files support Fluent .msh format or OpenFOAM polyMesh directories
- Output cases are saved to `run_chatcfd/` directory
- The system uses two LLM models: DeepSeek-R1 for reasoning tasks, DeepSeek-V3 for generation tasks
