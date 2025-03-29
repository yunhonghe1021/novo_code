# LucidLand

## Project Workflow

Follow these steps to run the entire pipeline:

### 1. Clone the Repository

```bash
cd LucidDreamer-Gaussian
```

### 2. Install Environment

Install the required Python packages:

```bash
pip install .
```

### 3. Launch Lucid Dreamer Frontend

Run the frontend app to upload and process images:

```bash
python app.py
```

- Use the interactive frontend to upload a **single image**.
- After processing, download the generated `.ply` file.

### 4. Run 4DGen Pipeline

Navigate to the `4DGen` directory and execute the pipeline:

```bash
cd ../4DGen
python complete-pipeline.py
```

This will generate a series of `.ply` files for visualization.

### 5. Visualize Results

Upload the generated `.ply` files to the online viewer:

- [SuperSpl.at Editor](https://superspl.at/editor)

## Requirements

Ensure you have Python 3.12 installed. Specific dependencies are handled during the installation step. We test that our H100 device need to run 30 mins and more.


