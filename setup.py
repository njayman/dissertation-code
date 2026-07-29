from setuptools import find_packages, setup

setup(
    name="devmind",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "devmind-train=devmind.trainer:main",
            "devmind-eval=devmind.evaluation:main",
            "devmind-gateway=devmind.gateway.app:main",
            "devmind-cloud=devmind.gateway.cloud_app:main",
        ],
    },
)
