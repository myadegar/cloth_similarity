# Cloth Similarity

## Requirements

```
python: >3.7, <3.11
numpy==1.22.0
```


GPU support:

First of all, you need to check if your system supports the `onnxruntime-gpu`.

## Usage as a command

'''
python cloth_similarity.py

'''
The command arguments:

    '-c', '--cloth_dir',   help= Input directory path of cloth images.
    '-p', '--person_dir',  help= Input directory path of person images.
    '-r', '--result_dir',  help= Output directory of result.
    '-x', '--cloth_mode',  help= segment part of body for similarity calculation (upper | lower)
    '-y', '--sim_method',  help= feature extraction method (reid | vae)
    '-z', '--sim_mode',    help= compare similarity mode between clothes and persons (mean | max)

