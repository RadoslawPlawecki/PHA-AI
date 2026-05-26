class DataProcessor:
    def __init__(self, df, config):
        self.df = df
        self.config = config

    def get_tool_dataframe(self):
        return self.df[self.config.tools]

    def get_long_dataframe(self):
        return self.df.melt(
            id_vars=["sample", "label"],
            value_vars=self.config.tools,
            var_name="tool",
            value_name="value"
        )
        