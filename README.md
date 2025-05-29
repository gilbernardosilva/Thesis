## Prerequisites for Valhalla and SUMO Setup

- Docker
- Docker Compose
- `wget` (optional, for downloading the data)
- Python / Pyenv
- SUMO (Simulation of Urban MObility)
- `unzip` (to handle `.gz` files, if needed)

## Setup Instructions

### 1. Install and Configure SUMO

1. **Install SUMO**:
   - Follow the official SUMO installation instructions for your operating system: [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing.html).
   - Ensure SUMO is installed correctly by running:
     ```bash
     sumo --version
     ```

2. **Set the `SUMO_HOME` Environment Variable**:
   - Define the `SUMO_HOME` environment variable to point to the SUMO installation directory (e.g., `/usr/share/sumo` or wherever SUMO is installed).
   - Add the following line to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`, or equivalent):
     ```bash
     export SUMO_HOME="/path/to/sumo/installation"
     ```
   - Replace `/path/to/sumo/installation` with the actual path to your SUMO installation.
   - Apply the changes:
     ```bash
     source ~/.bashrc  # Or ~/.zshrc, depending on your shell
     ```
   - Verify the variable is set:
     ```bash
     echo $SUMO_HOME
     ```

### 2. Download OSM Data

Download the latest Portugal OSM data from [Geofabrik](https://download.geofabrik.de/europe/portugal.html) or using the command below:

```bash
wget https://download.geofabrik.de/europe/portugal-latest.osm.pbf
```

### 3. Place the File in the Valhalla Data Directory

```bash
mkdir -p valhalla/data
mv portugal-latest.osm.pbf valhalla/data/
```

### 4. Run Valhalla with Docker Compose

```bash
docker compose up -d
```

It's normal that the initial setup (first time running) takes some time, as it is loading the graph and generating the necessary tiles. Ensure Valhalla is running in the background before proceeding.

### 5. Generate SUMO Network Files with OSM Web Wizard

To generate the `osm.net.xml.gz` and `osm.sumocfg` files for the region where your probes are located (e.g., Vila Nova de Gaia):

1. **Run the OSM Web Wizard**:
   - Use the following command to open the OSM Web Wizard and generate the network files:
     ```bash
     python $SUMO_HOME/tools/osmWebWizard.py -o ./sumo_output
     ```
   - This will open a browser window where you can select the region (e.g., Vila Nova de Gaia) and configure the simulation settings.
   - After completing the wizard, the files `osm.net.xml.gz` and `osm.sumocfg` will be saved in the `sumo_output` directory.

2. **Verify the Output Files**:
   - Ensure the `osm.net.xml.gz` and `osm.sumocfg` files are generated in the specified output directory.
   - If needed, unzip the `osm.net.xml.gz` file:
     ```bash
     gunzip sumo_output/osm.net.xml.gz
     ```

3. **Match File Suffixes**:
   - Ensure the suffix of the input files (e.g., probe CSV files) matches the variables defined in your pipeline configuration. For example, if your pipeline expects a file named `probes_gaia.csv`, the input file must have the same name and suffix.

### 6. Set Up the Python Virtual Environment and Install Requirements

```bash
pyenv local 3.11.8  # If not installed: pyenv install 3.11.8
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7. Prepare Probe Data

1. **Place Probe Files**:
   - Copy your probe data (e.g., CSV files) into the input directory expected by the pipeline.
   - Example:
     ```bash
     cp probes_gaia.csv input/
     ```

2. **Define Date Thresholds**:
   - Configure the date thresholds for the probe data in the pipeline configuration file (e.g., a settings file or environment variables).
   - Ensure the probe data corresponds to the region used in the SUMO network files (e.g., Vila Nova de Gaia).

### 8. Run the Valhalla Pipeline

Assuming Valhalla is running in the background, execute the pipeline:

```bash
python -m app.fcd.main
```

### 9. Run the SUMO Pipeline

After the Valhalla pipeline completes successfully, run the SUMO pipeline:

```bash
python -m app.sumo.main
```

## Notes

- Ensure the Valhalla container is running (`docker compose up -d`) before executing the pipelines.
- Verify that the probe data, SUMO network files, and pipeline configuration are consistent (e.g., same region, matching file names, and suffixes).
- If you encounter issues with the OSM Web Wizard, ensure `$SUMO_HOME` is correctly set and the SUMO tools are accessible.
