## train with prompt
1. A model that segment only 1 of N seismic facies (in **main.py**).
2. A model that segment one seismic facie, seeing all of the N facies, separatelly (in **_with_prompt/main.py**).
    - PS1: probablly this will works only with prompts.
    - PS2: apply augmentation in prompts:
        - use from 1 to 5 points (positive/negative)
        - use box