### Install

```
# 1. Create common folder and env for test
# ----------------------------------------
mkdir scatter_test
cd scatter_test
python3.12 -m venv venv
source venv/bin/activate

# 2. install imgui_bundle
# ----------------------------------------
git clone https://github.com/pthom/imgui_bundle.git
cd imgui_bundle
git checkout pyodide_oct24
pip install -v .     # this will take 10 minutes
cd ..

# 3. install fiatlight
# ----------------------------------------
git clone https://github.com/pthom/fiatlight.git
cd fiatlight
git checkout refact_io  # the code is hidden in a non standard branch
pip install -v -e .  # fast
cd ..

# 4. clone repo draft_probabl
# ----------------------------------------
git clone https://github.com/pthom/probabl_draft.git

# 5. install utilities
# ----------------------------------------
pip install scikit-learn jupyter


```

### Test and use

**python programs**

```
cd scatter
python xxx.py 
```

choose from
	scatter_bundle_app.py
	scatter_bundle_app_minimal.py
	scatter_fiatlight.py

**python notebooks:**
run `jupyter lab` and open scatter/scatter_notebook_fiatlight.ipynb

