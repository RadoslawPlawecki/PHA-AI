from .config import Config

from ..common.logger import (
    setup_logger,
    log_tqdm
)

from .io_utils import IOUtils
from .data_processor import DataProcessor

from .tool_analysis import ToolAnalysis
from .label_analysis import LabelAnalysis

from .plot_utils import PlotUtils


def main():
    logger = setup_logger("statistical_analysis")

    config = Config()
    io_utils = IOUtils(config)

    log_tqdm(logger, "[START] Loading data")

    df = io_utils.load_csv()

    processor = DataProcessor(df, config)

    tool_analysis = ToolAnalysis(
        processor,
        logger
    )

    descriptive = (
        tool_analysis.descriptive_statistics()
    )

    io_utils.save_dataframe(
        descriptive,
        "descriptive_statistics.csv"
    )

    normality = (
        tool_analysis.normality_tests()
    )

    io_utils.save_dataframe(
        normality,
        "normality_tests.csv"
    )

    friedman = (
        tool_analysis.friedman_test()
    )

    io_utils.save_dataframe(
        friedman,
        "friedman_test.csv"
    )

    anova = (
        tool_analysis.repeated_measures_anova()
    )

    io_utils.save_text(
        str(anova),
        "anova_results.txt"
    )

    posthoc = (
        tool_analysis.posthoc_tests()
    )

    io_utils.save_dataframe(
        posthoc,
        "posthoc_tests.csv"
    )

    label_analysis = LabelAnalysis(
        processor,
        logger
    )

    label_results = (
        label_analysis.run()
    )

    if label_results is not None:
        io_utils.save_dataframe(
            label_results,
            "label_analysis.csv"
        )

    PlotUtils(
        processor,
        io_utils,
        logger
    ).generate()

    log_tqdm(logger, "[FINISHED] Analysis completed")


if __name__ == "__main__":
    main()
