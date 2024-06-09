# Senior-Thesis_2024
Effects of Modulating Locus Coeruleus Noradrenergic Input to Cerebellar Interpositus Nucleus on Eyeblink Conditioning Performance

## Project Description
Briefly describe what this repository is for, what the project does, and what problems it aims to solve.

## Structure
This repository is organized into three main directories:

- **Data Processing**: Contains all the scripts and files necessary for data processing. This folder is essential for preparing the data before any analysis or operations are performed. Here are the main scripts in this directory:
  - `opto_ebc.py`: Processes and analyzes video data from EBC experiments, involving manual ROI selection and trace extraction to study eyeblink responses.
  - `lcopto.py`: Handles optogenetic stimulation trial data, expanding analysis to include additional trial conditions and generating comparative visualizations.
  - `blue_line.py`: Script for initial data preparation, including setting directory paths, validating file existence, and data integrity checks.
  - `regen_both.py`: Facilitates detailed analysis by selecting specific datasets for intensive processing, including video frame extraction and intensity normalization.

- **Batch Compilations**: Includes scripts that perform batch operations. These scripts are used for running processes that need to be executed in batches. Here are the main scripts in this directory:
  - `session_wise.py`: Compiles individual EBC session data to obtain consolidated average responses, filtering out erroneous data and expanding time course analysis.
  - `animal_wise.py`: Processes data on an individual animal basis, focusing on within-subject differences and excluding data from improperly prepared animals.
  - `batch_compile.py`: Aggregates data from eligible trials, compiling batch-level summaries and performing statistical comparisons between conditions.
  - `total_sessions_batched.py`: Analyzes data across different sessions systematically, focusing on session-by-session variability and the specific impacts of conditions.

- **Significance Calculations**: This folder contains all scripts and files used for calculating statistical significance of the results. Here are the main scripts in this directory:
  - `sig.py`: Calculates significance values.
  - `session_updated_sig.py`: Calculates all significance values on a session-wise basis.
  - `animal_updated_sig.py`: Calculates all significance values per animal.

## Built With
- Python 3.9.6 - Main programming language used.
- NumPy, pandas - Used for data handling and numerical operations.
- OpenCV - Used for image processing.
- Matplotlib - Used for generating visualizations.
- Jupyter Notebook - Used for scripting and documenting the analysis process.

## Authors
- **Thussenthan Walter-Angelo** - *Initial work* - [Thussenthan Walter-Angelo](https://github.com/thussenthanwalter-angelo)

## Acknowledgments
- Ben Deverett
- Samuel S.-H Wang Lab
- Department of Molecular Biology - Princeton University
- Princeton Neuroscience Institute - Princeton University
- Princeton University