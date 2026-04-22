param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

python training/train_grpo.py --config training/configs/pilot.yaml @ExtraArgs
