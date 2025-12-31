"""
LLM prompt templates for OpenFOAM case generation and correction.

These prompts are used by step functions to interact with the LLM.
"""

PROMPT_GENERATE_FILE = """\
Generate an OpenFOAM {file_path} file for the following case:

Case Description: {description}
Solver: {solver}
Turbulence Model: {turbulence_model}
Boundary Names: {boundaries}

Reference files from OpenFOAM tutorials:
{reference_files}

Requirements:
1. Generate ONLY the file content, no explanations
2. Use correct OpenFOAM syntax and formatting
3. Include appropriate boundary conditions for all boundaries listed
4. Ensure dimensions are correct for this solver type

Return the complete file content:"""

PROMPT_ANALYZE_ERROR = """\
Analyze this OpenFOAM runtime error to identify which file needs to be fixed:

Error message:
{error}

Case files:
{file_structure}

Respond with ONLY the file path (e.g., "system/fvSolution" or "0/U").
If multiple files might be involved, respond with the most likely one."""

PROMPT_CORRECT_FILE = """\
Fix this OpenFOAM file to resolve the runtime error.

Error message:
{error}

File to fix: {file_path}
Current content:
{current_content}

Reference files from tutorials:
{reference_files}

Boundary conditions available: {boundaries}

Requirements:
1. Return ONLY the corrected file content
2. No explanations or markdown formatting
3. Fix the specific issue causing the error
4. Maintain correct OpenFOAM syntax

Corrected file content:"""

PROMPT_REFLECT = """\
The OpenFOAM simulation keeps failing with similar errors despite corrections.

Current error:
{error}

Previous correction attempts:
{trajectory}

Case files:
{file_structure}

Please reflect on:
1. Why previous corrections didn't work
2. What might have been overlooked
3. What different approach should be tried

Provide a brief reflection and new strategy:"""

PROMPT_CHECK_NEW_FILE = """\
Analyze this OpenFOAM error to determine if a new file needs to be created:

Error:
{error}

If a file needs to be created, respond with the file path (e.g., "constant/g" or "0/nut").
If no new file is needed, respond with "no".

Response:"""

PROMPT_ADD_FILE = """\
Create a new OpenFOAM file: {file_path}

Case description: {description}
Solver: {solver}
Turbulence model: {turbulence_model}
Boundaries: {boundaries}

Existing case files for context:
{existing_files}

Reference files from tutorials:
{reference_files}

Requirements:
1. Return ONLY the file content
2. Use correct OpenFOAM syntax
3. Ensure compatibility with existing files

File content:"""

PROMPT_REWRITE_FILE = """\
Completely rewrite this OpenFOAM file from scratch.

The previous version had persistent errors that couldn't be fixed with patches.

File: {file_path}
Solver: {solver}
Turbulence model: {turbulence_model}
Boundaries: {boundaries}
Case description: {description}

Reference files from tutorials:
{reference_files}

Requirements:
1. Return ONLY the complete file content
2. Follow tutorial examples closely
3. Use correct dimensions and syntax

Complete file content:"""
