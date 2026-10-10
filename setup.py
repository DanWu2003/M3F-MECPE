"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


setup(
    name="m3f-mecpe",
    version="0.1.0",
    description="Multimodal emotion-cause pair extraction based on LAVIS and BLIP-2",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="multimodal, emotion cause pair extraction, BLIP-2, PyTorch",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="lavis.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.9",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)
