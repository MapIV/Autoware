from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    package_dir={
        "autoware_launcher_operator": "src/autoware_launcher_operator"},
    packages=[
        "autoware_launcher_operator",
        "autoware_launcher_operator.view",
        "autoware_launcher_operator.view.operations",
        "autoware_launcher_operator.view.plugins",
    ],
)

setup(**setup_args)
