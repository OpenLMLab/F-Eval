# Postprocess

After getting original results by OpenCompass, you can get the original results of each dataset in `results` folder.
Since the results of some subtasks are computed by different languages, we need to merge them into a single file by
running `merge_results.py`. Then we need to normalize them into the same scale by `normalize.py` to get the final score.

