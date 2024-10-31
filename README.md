# Some Are Never Right, Others Have Nothing Left: Unveiling Ideological Extremes in Parliamentary Debates Using BERT
### Abstract
The speeches in parliamentary debates can often outline the overall political situation in the countries. In this paper, we fine-tune BERT
on a downstream task of classifying political speeches by their political orientation. Afterward, we use representations of speeches to
compare political stances between the countries, as well as the difference between right and left policies within the specific country.
Because of the fine-tuning task, we can peek at the attention scores of the last layer to improve the understanding of what words are
associated with different ideologies in a political setting.
- ```dataset_creation_notebook.ipynb``` notebook for creating datasets from .tsv files
- ```training.ipynb``` notebook for fine tuning BERT
- ```pca-visual-bert.ipynb``` notebook for all plots
- ```attention-ins.ipynb``` notebook for attention analysis
