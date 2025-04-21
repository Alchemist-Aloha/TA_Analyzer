# Quick Start with Git and uv

This guide will help you get started with basic [Git](https://git-scm.com/) version control and Python dependency management using [uv](https://github.com/astral-sh/uv) for the TA_Analyzer project.

---

## 1. Basic Git Commands

- **Clone the repository:**

  ```sh
  git clone <repository-url>
  ```

- **Check status:**

  ```sh
  git status
  ```

- **Add changes:**

  ```sh
  git add <file>
  ```

- **Commit changes:**

  ```sh
  git commit -m "Describe your changes"
  ```

- **Push to GitHub:**

  ```sh
  git push
  ```

- **Pull latest changes:**

  ```sh
  git pull
  ```

- **Fetch latest changes:**
  ```sh
  git fetch
  ```

---

## 2. Cloning the Repository

If you haven't already, clone the repository from GitHub:

```sh
git clone https://github.com/Alchemist-Aloha/TA_Analyzer.git
```

This command creates a local copy of the repository on your machine.

If you plan to modify the code and contribute, consider forking the repository first:

1. Go to the [repository](https://github.com/Alchemist-Aloha/TA_Analyzer) on GitHub.
2. Click the "Fork" button in the top right corner.
3. Clone your forked repository:
   ```sh
   git clone <your-fork-url.git>
   ```
   Then navigate into the cloned directory:

```sh
cd TA_Analyzer
```

---

## 3. Setting Up Python Environment with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. You can use it as a drop-in replacement for pip.

### Install uv

If you don't have `uv` installed, you can install it with following commands in terminal:

macOS and Linux

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or with pip if you have python installed already:

```sh
pip install uv
```

### Create a Virtual Environment

It's recommended to use a virtual environment for Python projects:

```sh
uv venv --python 3.13
```

This command creates a virtual environment in the `.venv` directory.

### Activate the Virtual Environment

```sh
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate # Windows
```

This command activates the virtual environment. You should see the environment name in your terminal prompt.

Note that VSCode will automatically detect the virtual environment if you open the project folder in it.

### Deactivate the Virtual Environment

To deactivate the virtual environment, simply run:

```sh
deactivate
```

---

## 4. Installing Dependencies

Install all required dependencies from `requirements.txt`:

```sh
uv pip install -r requirements.txt
```

---

## 5. Updating Dependencies

To update all packages to the latest compatible versions:

```sh
uv pip install -r requirements.txt --upgrade
```

---

## 6. Running the Project

You can run the Jupyter notebooks .ipynb files directly in VSCode. Make sure you have IPYkernel installed in your virtual environment:

```sh
uv pip install ipykernel  # If not already installed
```

Then, you can open the Jupyter notebook in VSCode.

Or run Python scripts directly, for example:

```sh
uv run correction_line.py
```

For more details, see the [API documentation](https://alchemist-aloha.github.io/TA_Analyzer/) or the example notebook `ta_analyzer.ipynb`.
