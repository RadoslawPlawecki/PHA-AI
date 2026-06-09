import matplotlib.pyplot as plt
import seaborn as sns

from ..common.logger import log_tqdm


class PlotUtils:
    def __init__(self, processor, io_utils, logger):
        self.processor = processor
        self.io = io_utils
        self.logger = logger

    def generate(self):
        from ...plots.plots_formatting import use_latex
        use_latex()
        log_tqdm(self.logger, "[PLOTS] Generating plots")
        self._tool_boxplot()
        self._tool_violinplot()

    def _tool_boxplot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=self.processor.get_tool_dataframe(),
            ax=ax
        )
        ax.set_title(
            "Tool comparison"
        )
        self.io.save_plot(
            fig,
            "tool_boxplot.pdf"
        )

    def _tool_violinplot(self):
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )
        sns.violinplot(
            data=self.processor.get_long_dataframe(),
            x="tool",
            y="value",
            ax=ax
        )
        ax.set_title(
            "Tool distributions"
        )
        self.io.save_plot(
            fig,
            "tool_violinplot.pdf"
        )