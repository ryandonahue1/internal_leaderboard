from setuptools import find_packages, setup

setup(
    name="mlflow-header-plugin",
    version="0.0.1",
    description="Header plugin for MLflow",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={
        # Define a RequestHeaderProvider plugin. The entry point name for request header providers
        # is not used, and so is set to the string "unused" here
        "mlflow.request_header_provider": (
            "unused=mlflow_header_plugin.request_header_provider:PluginRequestHeaderProvider"
        ),
    },
)
