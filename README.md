# WhoSaidWhat
Open Source Code and Data for ICDL 2024 Paper: Who Said What? An Automated Approach to Analyzing Speech in Preschool Classrooms

Arxiv:https://arxiv.org/abs/2401.07342
IEEE ICDL 2024: https://ieeexplore.ieee.org/document/10644508

This project is designed to process and transcribe audio files using [OpenAI's Whisper](https://github.com/openai/whisper) model. It supports splitting audio into smaller segments, overlapping them, and transcribing the content.

## Whisper Part
### Features

- Load audio files from a specified directory (supports `.wav` and `.mp3` formats).
- Process audio by splitting it into segments.
- Optionally overlap segments to avoid data loss at boundaries.
- Use different Whisper model sizes (`large-v3`, `large-v2`) for transcription.
- Save transcription results for further analysis.

### Requirements

Before running the script, ensure you have installed all required dependencies by setting up the Conda environment using the provided `whisper_env.yml` file.

#### Install Whisper Conda Environment

1. Create the environment using `whisper_env.yml`:

   ```bash
   conda env create -f whisper_env.yml
   ```

2. Activate the environment:

   ```bash
   conda activate whisper_env
   ```

### Usage

#### Command-line Interface

You can run the script directly from the command line. Below are the available options and their descriptions:

```bash
python WhisperTrans.py [audio_file_pth] [output_file_pth] [--cut_minutes CUT_MINUTES] [--overlap_minutes OVERLAP_MINUTES] [--model_name MODEL_NAME]
```

#### Arguments

- `audio_file_pth` (str): The directory containing the audio files to process.
- `output_file_pth` (str): The directory where the transcription results will be saved.
- `--cut_minutes` (float, optional): Length of each audio segment in minutes (default: 2 minutes).
- `--overlap_minutes` (int, optional): Overlap duration between consecutive segments in minutes (default: 0 minutes).
- `--model_name` (str, optional): Whisper model to use (`large-v3` or `large-v2`, default: `large-v3`).

#### Example

To process audio files in the `audio/` directory, split them into 2-minute segments with 30 seconds of overlap, and save the results to the `output/` directory:

```bash
python WhisperTrans.py audio/ output/ --cut_minutes 2 --overlap_minutes 0.5 --model_name large-v3
```

### Functions

#### `save_list_to_file(lst, filename)`

Saves a list to a specified file using Python's `pickle` module.

#### `process_AST(audio_file_pth, output_file_pth, cut_minutes, overlap_minutes, model_name)`

Main function for processing audio files. It loads the specified Whisper model, splits the audio into segments, and performs transcription on each segment.

---

## Alice Part

Follow the instructions via: https://github.com/orasanen/ALICE

---
