import os
import pandas as pd
import matplotlib.pyplot as plt


class IOUtils:
    def __init__(self, config):
        self.config = config
        os.makedirs(
            self.config.output_dir,
            exist_ok=True
        )

    def load_csv(self):
        return pd.read_csv(self.config.in_file)

    def save_dataframe(self, df, filename):
        path = os.path.join(self.config.output_dir, filename)
        df.to_csv(path, index=False)

    def save_text(self, text, filename):
        path = os.path.join(
            self.config.output_dir,
            filename
        )
        with open(path, "w") as f:
            f.write(text)

    def save_plot(self, fig, filename):
        path = os.path.join(self.config.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        