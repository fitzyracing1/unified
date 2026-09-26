# unified

Backwards AI. Specify the output first. Derive the model. Freeze a forward runtime.

You do not start with data, architecture, or a prompt.

You start with the thing the system must emit.

## Order of work (fixed)

1. Write the output contract.
2. Pin examples of the finished answer.
3. Derive the smallest function that can emit those answers.
4. Freeze a forward runtime that no longer contains the contract text.

Training data is a side effect of locking the output, not the starting material.

## Files

- `backwards_ai.py` — contract, builder, nearest-prototype runtime
- `runtime.json` — frozen forward model after `python3 backwards_ai.py`
- `cli.py` — query the frozen model

## Run

```
python3 backwards_ai.py
python3 cli.py predict '{"income":48000,"years_local":6,"prior_defaults":0,"family_tie":true}'
```

Fit on the seed contract is 1.0: every specified output is reconstructed.

## Next increment

Swap nearest-prototype for a fit linear map when the schema is float and examples > features.
