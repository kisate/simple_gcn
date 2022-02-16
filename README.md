# Simple GCN

Test project for JetBrains Research internship.

***

## Installation

* Install requirements (Anaconda may have different package names):

```commandline
pip install -r requirements.txt
```

* Install pytorch-geometric:

You can use pip (it should work)

```commandline
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-geometric
```

In case you use Anaconda, there is a simpler way:

```commandline
conda install pyg -c pyg -c conda-forge
```

If installation fails please refer to the pytorch-geometric [installation docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), you probably need to install torch-scatter and torch-sparse for another cuda version.
