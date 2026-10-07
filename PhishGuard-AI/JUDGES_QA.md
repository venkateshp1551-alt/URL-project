# Judges' Q&A Preparation

> Preparation prompts only. Replace planned-work descriptions with verified
> implementation details as the project is built.

## What problem does the project address?

It aims to help identify potentially phishing URLs before users interact with
them.

## How does the detection approach work?

The planned approach is to prepare URL examples, train a Scikit-learn model,
evaluate it, and expose predictions through a Flask backend. The model has not
been implemented yet.

## How will you measure performance?

Use a held-out evaluation set and report appropriate classification metrics,
including precision, recall, F1 score, and a confusion matrix. Report actual
results only after evaluation.

## What are the main limitations?

Detection quality will depend on the dataset and may not generalize to new or
changing phishing techniques. Predictions should be treated as a warning, not
as a guarantee that a URL is safe.

## What would you improve next?

Expand and refresh the data, test on data from a different source or time
period, inspect false positives and negatives, and improve the user experience.
