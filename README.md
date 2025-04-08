# Princeton University Senior Thesis (2024)
Title: **Effects of Modulating Locus Coeruleus Noradrenergic Input to Cerebellar Interpositus Nucleus on Eyeblink Conditioning Performance** 

By Thussenthan Walter-Angelo ('24)

## Project Description
This repository contains the code developed for my senior thesis, “Effects of Modulating Locus Coeruleus Noradrenergic Input to Cerebellar Interpositus Nucleus on Eyeblink Conditioning Performance.” The project investigates how optogenetically mediated modulation of noradrenergic signaling from the locus coeruleus (LC) to the cerebellar interpositus nucleus (INT) affects associative learning, as measured by eyeblink conditioning (EBC).

Key components include:

- EBC Training and Optogenetic Stimulation: Code modules (lcopto.py and opto_ebc.py) are used to control and monitor the classical conditioning paradigm and the delivery of timed optogenetic stimuli.
- Data Analysis Pipelines: Scripts such as batch_compile.py, animal_wise.py, and session_wise.py process behavioral data, extracting metrics like the conditioned response (CR) amplitude, timing, and rate of rise. These analyses assess the impact of different stimulation durations (one-second versus two-second protocols) on learning performance.
- Research Insights: The project aims to elucidate the temporal dynamics of LC–INT interactions, contributing to our understanding of neuromodulation in learning and memory. Findings suggest that precise, time-dependent activation of LC axons can significantly alter the expression of learned behaviors.

This codebase, along with detailed documentation and accompanying data, offers a resource for researchers interested in neurobiological mechanisms of learning, optogenetic interventions, and advanced behavioral analysis.

### Structure
This repository is organized into three main directories:

- **Data Processing**: Contains all the scripts and files necessary for data processing. This folder is essential for preparing the data before any analysis or operations are performed. Here are the main scripts in this directory:
  - `opto_ebc.py`: Processes and analyzes video data from EBC experiments, involving manual ROI selection and trace extraction to study eyeblink responses.
  - `lcopto.py`: Handles optogenetic stimulation trial data, expanding the analysis to include additional trial conditions and generating comparative visualizations.
  - `blue_line.py`: Script for initial data preparation, including setting directory paths, validating file existence, and data integrity checks.
  - `regen_both.py`: Facilitates detailed analysis by selecting specific datasets for intensive processing, including video frame extraction and intensity normalization.

- **Batch Compilations**: Includes scripts that perform batch operations. These scripts are used for running processes that need to be executed in batches. Here are the main scripts in this directory:
  - `session_wise.py`: Compiles individual EBC session data to obtain consolidated average responses, filtering out erroneous data and expanding time course analysis.
  - `animal_wise.py`: Processes data on an individual animal basis, focusing on within-subject differences and excluding data from improperly prepared animals.
  - `batch_compile.py`: Aggregates data from eligible trials, compiling batch-level summaries and performing statistical comparisons between conditions.
  - `total_sessions_batched.py`: Analyzes data across different sessions systematically, focusing on session-by-session variability and the specific impacts of conditions.

- **Significance Calculations**: This folder contains all scripts and files used for calculating the statistical significance of the results. Here are the main scripts in this directory:
  - `sig.py`: Calculates significance values.
  - `session_updated_sig.py`: Calculates all significance values on a session-wise basis.
  - `animal_updated_sig.py`: Calculates all significance values per animal.

### Built With
- Python 3.9.6 - Main programming language used.
- NumPy, pandas - Used for data handling and numerical operations.
- OpenCV - Used for image processing.
- Matplotlib - Used for generating visualizations.
- Jupyter Notebook - Used for scripting and documenting the analysis process.

## Authors
- **Thussenthan Walter-Angelo** - *Initial work*
- If you require additional information on the analysis/data, please contact [Thussenthan Walter-Angelo](t.walterangelo@gmail.com).

## Acknowledgments
- Dr. Ben Deverett
- Dr. Junuk Lee
- Dr. Gerard Joey Broussard
- Dr. Samuel S.-H. Wang
- All the lab members of the Wang Lab
- Department of Molecular Biology - Princeton University
- Princeton Neuroscience Institute - Princeton University
- Princeton University
