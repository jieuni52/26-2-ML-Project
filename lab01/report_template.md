> Fill in this template, then export to **report.pdf** for submission
> (e.g., VS Code/Typora export, or `pandoc report.md -o report.pdf`).
>
> **Budget**: 0.5–1 page of text **plus** Figures 1–3, ≤ 3 pages total. Text-based PDF, not a scan.
>
> **Caption format is the same as the concept note**: number, a short title, then one or two
> sentences of explanation. Table captions go **above** the table, figure captions **below** the
> figure. Every figure and table must be referred to by number somewhere in your text — an
> unreferenced figure earns nothing.
>
> Delete these instruction lines before exporting.

# Lab 1 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton: a deviation, an extension, or a
decision you had to make.)

## 2. Results

**Table 1. Key numbers from `results.json`.** (One sentence: which of these you did not expect,
or which one your interpretation in §4 hangs on.)

| Metric | Value |
|--------|-------|
| Rows raw / clean |  |
| Duplicates removed |  |
| Missing values (raw → clean) |  |
| Quantity outliers (IQR) |  |
| Top category by revenue |  |
| Mean rating (clean) |  |

## 3. Figures

(Export from `notebooks/lab01_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. Missing values per column.** (One or two sentences: which columns fail, and what that
suggests about how this data was collected.)

{image}

**Figure 2. Distribution of `total_price` after cleaning.** (One or two sentences: the shape of the
distribution, and what it implies about summarizing this column with a mean or a median.)

{image}

**Figure 3. `customer_rating` by category after cleaning.** (One or two sentences: which category
stands out, and whether its group size makes that difference trustworthy.)

## 4. Interpretation

(Why do the numbers and figures look this way? Refer to figures and the table by number. For
example: what do the flagged outliers actually represent in this data, and is removing them
justified? Which imputation choice could bias which statistic, and can you see that bias in
Figure 2? Does the top category in Table 1 survive a look at its count?)

## 5. Limitations

(One concrete thing that would change your conclusions.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
