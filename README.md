# 3D4D: An Interactive, Editable, 4D World Model via 3D Video Generation

3D4D is an interactive 4D visualization framework that integrates WebGL with Supersplat rendering. It transforms static images and text into 4D scenes through four core modules and employs a foveated rendering strategy for efficient, real-time multi-modal interaction.


## Project Workflow
![demo1.png](demo1.png)

![demo2.png](demo2.png)
## Usage

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

```bash
cd ../supersplat
npm install
npm run develop
```

Navigate to ```http://localhost:3000```

## Requirements

Ensure you have Python 3.12 installed. Specific dependencies are handled during the installation step. We test that our H100 device need to run 30 mins and more.


