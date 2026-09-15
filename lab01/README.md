# Lab 1 — Reproduction Instructions

## Environment

- Python 3.10.11

## Commands

Run the following commands from the Lab 1 project root:

```bash
python -m pip install -r requirements.txt
python src/lab01.py
python -m pytest tests/ -q
```

`python src/lab01.py` runs the complete data-processing pipeline and regenerates
`results.json`. The final command runs the six public tests.
