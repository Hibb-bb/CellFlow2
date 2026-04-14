import pandas as pd

# url = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/DE_stats.suppl_table.csv"
df = pd.read_csv("/projects/b1094/ywl7940/CellFlow2/docs/notebooks/DE_stats.suppl_table.csv", usecols=["target_contrast_gene_name"])
genes = df["target_contrast_gene_name"].unique()
print(genes)
print(f"\nTotal unique perturbed genes: {len(genes)}")